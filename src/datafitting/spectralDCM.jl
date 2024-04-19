"""
spectralDCM.jl

Main functions to compute a spectral DCM.

transferfunction : computes transfer function of neuronal model as well as measurement model
csd_approx       : approximates CSD based on transfer functions
csd_fmri_mtf     :
diff             : computes Jacobian of model
csd_Q            : computes precision component prior (which erroneously is not used in the SPM12 code for fMRI signals, it is used for other modalities)
matlab_norm      : computes norm consistent with MATLAB's norm function (Julia's is different, at lest for matrices. Haven't tested vectors)
spm_logdet       : mimick SPM12's way to compute the logarithm of the determinant. Sometimes Julia's logdet won't work.
variationalbayes : main routine that computes the variational Bayes estimate of model parameters
"""

tagtype(::Dual{T,V,N}) where {T,V,N} = T

# struct types for Variational Laplace
mutable struct VLState
    iter::Int                    # number of iteration
    v::Float64                   # log ascent rate of SPM style Levenberg-Marquardt optimization
    F::Vector{Float64}           # free energy vector (store at each iteration)
    dF::Vector{Float64}          # predicted free energy changes (store at each iteration)
    λ::Vector{Float64}           # hyperparameter
    ϵ_θ::Vector{Float64}         # prediction error of parameters θ
    reset_state::Vector{Any}     # store state to reset to [ϵ_θ and λ] when the free energy deteriorates
    μθ_po::Vector{Float64}       # posterior expectation value of parameters 
    Σθ_po::Matrix{Float64}       # posterior covariance matrix of parameters
    dFdθ::Vector{Float64}        # free energy gradient w.r.t. parameters
    dFdθθ::Matrix{Float64}       # free energy Hessian w.r.t. parameters
end

struct VLSetup{Model, N}
    model_at_x0::Model                        # model evaluated at initial conditions
    y_csd::Array{ComplexF64, N}                 # cross-spectral density approximated by fitting MARs to data
    tolerance::Float64                        # convergence criterion
    systemnums::Vector{Int}                   # several integers -> np: n. parameters, ny: n. datapoints, nq: n. Q matrices, nh: n. hyperparameters
    systemvecs::Vector{Vector{Float64}}       # μθ_pr: prior expectation values of parameters and μλ_pr: prior expectation values of hyperparameters
    systemmatrices::Vector{Matrix{Float64}}   # Πθ_pr: prior precision matrix of parameters, Πλ_pr: prior precision matrix of hyperparameters
    Q::Matrix{ComplexF64}                     # linear decomposition of precision matrix of parameters, typically just one matrix, the empirical correlation matrix
end

"""
    function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    
    Dispatch of LinearAlgebra.eigen for dual matrices with complex numbers. Make the eigenvalue decomposition 
    amenable to automatic differentiation. To do so compute the analytical derivative of eigenvalues
    and eigenvectors. 

    Arguments:
    - `M`: matrix of type Dual of which to compute the eigenvalue decomposition. 

    Returns:
    - `Eigen(evals, evecs)`: eigenvalue decomposition returned as type LinearAlgebra.Eigen

"""
function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    nd = size(M, 1)
    A = (p->p.value).(M)
    F = eigen(A, sortby=nothing, permute=true)
    λ, V = F
    local ∂λ_agg, ∂V_agg
    # compute eigenvalue and eigenvector derivatives for all partials
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)   # copy only needed for Complex because `real.(v)` makes a new array
        ∂K ./= transpose(λ) .- λ
        fill!(∂Kdiag, 0)
        ∂V_tmp = mul!(tmp, V, ∂K)
        _eigen_norm_phase_fwd!(∂V_tmp, A, V)
        if i == 1
            ∂V_agg = ∂V_tmp
            ∂λ_agg = ∂λ_tmp
        else
            ∂V_agg = cat(∂V_agg, ∂V_tmp, dims=3)
            ∂λ_agg = cat(∂λ_agg, ∂λ_tmp, dims=2)
        end
    end
    # reassemble the aggregated vectors and values into a Partials type
    ∂V = map(Iterators.product(1:nd, 1:nd)) do (i, j)
        Partials(NTuple{np}(∂V_agg[i, j, :]))
    end
    ∂λ = map(1:nd) do i
        Partials(NTuple{np}(∂λ_agg[i, :]))
    end
    if eltype(V) <: Complex
        evals = map(λ, ∂λ) do x, y
            rex, imx = reim(x)
            rey, imy = real.(Tuple(y)), imag.(Tuple(y))
            Complex(Dual{T}(rex, Partials(rey)), Dual{T}(imx, Partials(imy)))
        end
        evecs = map(V, ∂V) do x, y
            rex, imx = reim(x)
            rey, imy = real.(Tuple(y)), imag.(Tuple(y))
            Complex(Dual{T}(rex, Partials(rey)), Dual{T}(imx, Partials(imy)))
        end
    else
        evals = Dual{T}.(λ, ∂λ)
        evecs = Dual{T}.(V, ∂V)
    end
    return Eigen(evals, evecs)
end

function transferfunction_fmri(ω, derivatives, params, params_idx)
    ∂f = derivatives(params[params_idx[:dspars]])
    idx_ds = deleteat!([1:size(∂f, 1);], sort(vcat(params_idx[:bold], params_idx[:u])))
    ∂f∂x = ∂f[idx_ds, idx_ds]
    ∂f∂u = ∂f[idx_ds, params_idx[:u]]
    ∂g∂x = ∂f[params_idx[:bold], idx_ds]

    F = eigen(∂f∂x)
    Λ = F.values
    V = F.vectors

    ∂g∂v = ∂g∂x*V
    ∂v∂u = V\∂f∂u              # u is external variable which we don't use right now. With external variable this would read V/dfdu

    nω = size(ω, 1)            # number of frequencies
    ng = size(∂g∂x, 1)         # number of outputs
    nu = size(∂v∂u, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(∂v∂u))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                S[:,i,j] .+= (∂g∂v[i,k]*∂v∂u[k,j]) .* ((1im*2*pi) .* ω .- Λ[k]).^-1 
            end
        end
    end

    return S
end


"""
    This function implements equation 2 of the spectral DCM paper, Friston et al. 2014 "A DCM for resting state fMRI".
    Note that nomenclature is taken from SPM12 code and it does not seem to coincide with the spectral DCM paper's nomenclature. 
    For instance, Gu should represent the spectral component due to external input according to the paper. However, in the code this represents
    the hidden state fluctuations (which are called Gν in the paper).
    Gn in the code corresponds to Ge in the paper, i.e. the observation noise. In the code global and local components are defined, no such distinction
    is discussed in the paper. In fact the parameter γ, corresponding to local component is not present in the paper.
"""
function csd_approx(ω, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nω = length(ω)
    nd = length(params_idx[:lnγ])
    α = params[params_idx[:lnα]]
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]
    
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    G = ω.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nω, nd, nd)
    Gn = zeros(eltype(G), nω, nd, nd)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1]) .* G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = ω.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nd
        Gn[:,i,i] .+= exp(γ[i])*G
    end

    # global components
    for i = 1:nd
        for j = i:nd
            Gn[:,i,j] .+= exp(β[1])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    S = transferfunction_fmri(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end

@views function csd_fmri_mtf(freqs, p, derivatives, params, params_idx)   # alongside the above realtes to spm_csd_fmri_mtf.m
    G = csd_approx(freqs, derivatives, params, params_idx)
    dt = 1/(2*freqs[end])
    # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
    # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
    # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. Friston conferms that likely it is
    # to make y well behaved.
    mar = csd2mar(G, freqs, dt, p-1)
    y = mar2csd(mar, freqs)
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    return y
end

"""
    function matlab_norm(A, p)

    Simple helper function to implement the norm of a matrix that is equivalent to the one given in MATLAB for order=1, 2, Inf. 
    This is needed for the reproduction of the exact same results of SPM12.

    Arguments:
    - `A`: matrix
    - `p`: order of norm
"""
function matlab_norm(M, p)
    if p == 1
        return maximum(sum(abs, M, dims=1))
    elseif p == Inf
        return maximum(sum(abs, M, dims=2))
    elseif p == 2
        print("Not implemented yet!\n")
        return NaN
    end
end

function csd_Q(csd)
    s = size(csd)
    Qn = length(csd)
    Q = zeros(ComplexF64, Qn, Qn);
    idx = CartesianIndices(csd)
    for Qi  = 1:Qn
        for Qj = 1:Qn
            if idx[Qi][1] == idx[Qj][1]
                Q[Qi,Qj] = csd[idx[Qi][1], idx[Qi][2], idx[Qj][2]]*csd[idx[Qi][1], idx[Qi][3], idx[Qj][3]]
            end
        end
    end
    Q = inv(Q + matlab_norm(Q, 1)*I/32)   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
end

"""
    function spm_logdet(M)

    SPM12 style implementation of the logarithm of the determinant of a matrix.

    Arguments:
    - `M`: matrix
"""
function spm_logdet(M)
    TOL = 1e-16
    s = diag(M)
    if sum(abs, s) != sum(abs, M)
        s = svdvals(M)
    end
    return sum((log(sval) for sval in s if TOL < sval < inv(TOL)), init=zero(eltype(s))) 
end

"""
    vecparam(param::OrderedDict)

    Function to flatten an ordered dictionary of model parameters and return a simple list of parameter values.

    Arguments:
    - `param`: dictionary of model parameters (may contain numbers and lists of numbers)
"""
function vecparam(param::OrderedDict)
    flatparam = Float64[]
    for v in values(param)
        if v isa Array
            for vv in v
                push!(flatparam, vv)
            end
        else
            push!(flatparam, v)
        end
    end
    return flatparam
end


"""
    function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, params_idx)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

    Arguments:
    - `data`        : dataframe with column names corresponding to the regions of measurement.
    - `model`       : MTK model, including state evolution and measurement.
    - `initcond`    : dictionary of initial conditions, numerical values for all states
    - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
    -- `dt`         : sampling interval
    -- `freq`       : frequencies at which to evaluate the CSD
    -- `p`          : order parameter of the multivariate autoregression model
    - `priors`      : dataframe of parameters with the following columns:
    -- `name`       : corresponds to MTK model name
    -- `mean`       : corresponds to prior mean value
    -- `variance`   : corresponds to the prior variances
    - `hyperpriors` : dataframe of parameters with the following columns:
    -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
    - `params_idx`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
"""
function setup_sDCM(data, model, initcond, csdsetup, priors, hyperpriors, params_idx)
    # compute cross-spectral density
    dt = csdsetup[:dt];              # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    ω = csdsetup[:freq];             # frequencies at which the CSD is evaluated
    p = csdsetup[:p];                # order of MAR
    mar = mar_ml(Matrix(data), p);   # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, ω, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
    jac_fg = generate_jacobian(model, expression = Val{false})[1]   # compute symbolic jacobian.

    statevals = [v for v in values(initcond)]
    derivatives = par -> jac_fg(statevals, addnontunableparams(par, model), t)

    μθ_pr = vecparam(OrderedDict(priors.name .=> priors.mean))            # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = diagm(vecparam(OrderedDict(priors.name .=> priors.variance)))

    ### Collect prior means and covariances ###
    Q = csd_Q(y_csd);                 # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)

    f = params -> csd_fmri_mtf(ω, p, derivatives, params, params_idx)

    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLState(
        0,             # iter
        -4,            # log ascent rate
        [-Inf],        # free energy
        Float64[],            # delta free energy
        8*ones(nh),    # metaparameter, initial condition. TODO: why are we not just using the prior mean?
        zeros(np),     # parameter estimation error ϵ_θ
        [zeros(np), 8*ones(nh)],      # memorize reset state
        μθ_pr,         # parameter posterior mean
        Σθ_pr,         # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )
    # variational laplace setup
    vlsetup = VLSetup(
        f,                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                # empirical cross-spectral density
        1e-1,                 # tolerance
        [np, ny, nq, nh],     # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, hyperpriors[:μλ_pr]],          # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors[:Πλ_pr]],     # parameter and hyperparameter prior precision matrices
        Q                                      # components of data precision matrix
    )
    return (vlstate, vlsetup)
end

function run_sDCM_iteration!(state::VLState, setup::VLSetup)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)

    norm_dfdp = matlab_norm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);
            t = exp(v - logdet(dFdθθ)/np)

            # E-Step: update
            if t > exp(16)
                ϵ_θ = ϵ_θ - dFdθθ \ dFdθ    # -inv(dfdx)*f
            else
                ϵ_θ = ϵ_θ + expv(t, dFdθθ, dFdθθ \ dFdθ) - dFdθθ \ dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
            end

            μθ_po = μθ_pr + ϵ_θ

            dfdp = jacobian(f, μθ_po)

            # check for stability
            norm_dfdp = matlab_norm(dfdp, Inf);
            revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f(μθ_po), ny)                   # error
    J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(eltype(J), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        for i = 1:nh
            P[:,:,i] = Q[:,:,i]*exp(λ[i])
            PΣ[:,:,i] = iΣ \ P[:,:,i]
            JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
        end
        for i = 1:nh
            dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
            for j = i:nh
                dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                dFdλλ[j, i] = dFdλλ[i, j]
            end
        end

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        t = exp(4 - spm_logdet(dFdλλ)/length(λ))
        # E-Step: update
        if t > exp(16)
            dλ = -real(dFdλλ \ dFdλ)
        else
            idFdλλ = inv(dFdλλ)
            dλ = real(exponential!(t * dFdλλ) * idFdλλ*dFdλ - idFdλλ*dFdλ)   # (expm(dfdx*t) - I)*inv(dfdx)*f ~~~ could also be done with expv but doesn't work with Dual.
        end

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    t = exp(v - spm_logdet(dFdθθ)/np)
    if t > exp(16)
        dθ = - inv(dFdθθ) * dFdθ     # -inv(dfdx)*f
    else
        dθ = exponential!(t * dFdθθ) * inv(dFdθθ) * dFdθ - inv(dFdθθ) * dFdθ     # (expm(dfdx*t) - I)*inv(dfdx)*f
    end

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end
