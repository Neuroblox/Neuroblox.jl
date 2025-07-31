"""
spectralDCM.jl

Main functions to compute a spectral DCM.

transferfunction : computes transfer function of neuronal model as well as measurement model
csd_approx       : approximates CSD based on transfer functions
csd_fmri_mtf     :
diff             : computes Jacobian of model
csd_Q            : computes precision component prior (which erroneously is not used in the SPM12 code for fMRI signals, it is used for other modalities)
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
    reset_state::Vector{Any}     # store state to reset to [ϵ_θ and λ] when the free energy gets worse rather than better
    μθ_po::Vector{Float64}       # posterior expectation value of parameters 
    Σθ_po::Matrix{Float64}       # posterior covariance matrix of parameters
    dFdθ::Vector{Float64}        # free energy gradient w.r.t. parameters
    dFdθθ::Matrix{Float64}       # free energy Hessian w.r.t. parameters
end

struct VLSetup{Model, T1 <: Array{ComplexF64}, T2 <: AbstractArray}
    model_at_x0::Model                        # model evaluated at initial conditions
    y_csd::T1                                 # cross-spectral density approximated by fitting MARs to data
    tolerance::Float64                        # convergence criterion
    systemnums::Vector{Int}                   # several integers -> np: n. parameters, ny: n. datapoints, nq: n. Q matrices, nh: n. hyperparameters
    systemvecs::Vector{Vector{Float64}}       # μθ_pr: prior expectation values of parameters and μλ_pr: prior expectation values of hyperparameters
    systemmatrices::Vector{Matrix{Float64}}   # Πθ_pr: prior precision matrix of parameters, Πλ_pr: prior precision matrix of hyperparameters
    Q::T2                                     # linear decomposition of precision matrix of parameters, typically just one matrix, the empirical correlation matrix
    modelparam::OrderedDict
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

function transferfunction(freq, derivatives, params, indices)
    ∂f = derivatives(params[indices[:dspars]])
    ∂f∂x = ∂f[indices[:sts], indices[:sts]]
    ∂f∂u = ∂f[indices[:sts], indices[:u]]
    ∂g∂x = ∂f[indices[:m], indices[:sts]]

    F = eigen(∂f∂x)
    Λ = F.values
    V = F.vectors

    ∂g∂v = ∂g∂x*V
    ∂v∂u = V\∂f∂u              # u is external variable which we don't use right now. With external variable this would read V/dfdu

    nfreq = size(freq, 1)      # number of frequencies
    ng = size(∂g∂x, 1)         # number of outputs
    nu = size(∂v∂u, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(∂v∂u))}, nfreq, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                S[:,i,j] .+= (∂g∂v[i,k]*∂v∂u[k,j]) .* ((1im*2*pi) .* freq .- Λ[k]).^-1 
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
function csd_approx(freq, derivatives, params, indices)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nfreq = length(freq)
    nrr = length(indices[:lnγ])
    α = params[indices[:lnα]]
    β = params[indices[:lnβ]]
    γ = params[indices[:lnγ]]
    
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    G = freq.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nfreq, nrr, nrr)
    Gn = zeros(eltype(G), nfreq, nrr, nrr)
    for i = 1:nrr
        Gu[:, i, i] .+= exp(α[1]) .* G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = freq.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nrr
        Gn[:,i,i] .+= exp(γ[i])*G
    end

    # global components
    for i = 1:nrr
        for j = i:nrr
            Gn[:,i,j] .+= exp(β[1])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    S = transferfunction(freq, derivatives, params, indices)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nfreq, nrr, nrr);
    for i = 1:nfreq
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end

function csd_approx_lfp(freq, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nfreq = length(freq)
    nrr = length(params_idx[:lnγ])
    α = reshape(params[params_idx[:lnα]], nrr, nrr)
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]

    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".
    Gu = zeros(eltype(α), nfreq, nrr)   # spectrum of neuronal innovations or intrinsic noise or system noise
    Gn = zeros(eltype(β), nfreq)   # global spectrum of channel noise or observation noise or external noise
    Gs = zeros(eltype(γ), nfreq)   # region specific spectrum of channel noise or observation noise or external noise
    for i = 1:nrr
        Gu[:, i] .+= exp(α[1, i]) .* freq.^(-exp(α[2, i]))
    end
    # global components and region specific observation noise (1/f or AR(1) form)
    Gn = exp(β[1] - 2) * freq.^(-exp(β[2]))
    Gs = exp(γ[1] - 2) * freq.^(-exp(γ[2]))  # this is really oddly implemented in SPM12. Completely unclear how this should be region specific

    S = transferfunction(freq, derivatives, params, params_idx)   # This is K(freq) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nfreq, nrr, nrr);
    for i = 1:nfreq
        G[i,:,:] = S[i,:,:]*diagm(Gu[i,:])*S[i,:,:]'
    end

    for i = 1:nrr
        G[:,i,i] += Gs
        for j = 1:nrr
            G[:,i,j] += Gn
        end
    end

    return G
end

function csd_mtf(freq, p::Int64, derivatives, params, params_idx, modality::String)   # alongside the above realtes to spm_csd_fmri_mtf.m
    if modality == "fMRI"
        G = csd_approx(freq, derivatives, params, params_idx)
        dt = 1/(2*freq[end])
        # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
        # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
        # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. NB: Friston conferms that likely it is
        # to make y well behaved.
        mar = csd2mar(G, freq, dt, p-1)
        y = mar2csd(mar, freq)
    elseif modality == "LFP"
        y = csd_approx_lfp(freq, derivatives, params, params_idx)
    end
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    return y
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
    Q = inv(Q + opnorm(Q, 1)*I/32)
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

function integration_step(dfdx, f, v, solenoid=false)
    if solenoid
        # add solenoidal mixing as is present in the later versions of SPM, in particular SPM25
        L  = tril(dfdx);
        Q  = L - L';
        Q  = Q/opnorm(Q, 2)/8;

        f  = f  - Q*f;
        dfdx = dfdx - Q*dfdx;        
    end

    # NB: (exp(dfdx*t) - I)*inv(dfdx)*f, could also be done with expv (expv(t, dFdθθ, dFdθθ \ dFdθ) - dFdθθ \ dFdθ) but doesn't work with Dual.
    # Could also be done with `exponential!` but isn't numerically stable.
    # Thus, just use `exp`.
    n = length(f)
    t = exp(v - spm_logdet(dfdx)/n)

    if t > exp(16)
        dx = - dfdx \ f   # -inv(dfdx)*f
    else
        dx = (exp(t * dfdx) - I) * inv(dfdx) * f # (expm(dfdx*t) - I)*inv(dfdx)*f
    end

    return dx
end

function defaultprior(model, nrr)
    _, idx_sts = get_dynamic_states(model)
    idx_u = get_idx_tagged_vars(model, "ext_input")                  # get index of external input state
    idx_bold, _ = get_eqidx_tagged_vars(model, "measurement")  # get index of equation of bold state

    # collect parameter default values, these constitute the prior mean.
    parammean = OrderedDict()
    np = sum(tunable_parameters(model); init=0) do par
        val = Symbolics.getdefaultval(par)
        parammean[par] = val
        length(val)
    end
    indices = Dict(:dspars => collect(1:np))
    # Noise parameters
    parammean[:lnα] = [0.0, 0.0];            # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
    n = length(parammean[:lnα]);
    indices[:lnα] = collect(np+1:np+n);
    np += n;
    parammean[:lnβ] = [0.0, 0.0];            # global observation noise, ln(β) as above
    n = length(parammean[:lnβ]);
    indices[:lnβ] = collect(np+1:np+n);
    np += n;
    parammean[:lnγ] = zeros(Float64, nrr);   # region specific observation noise
    indices[:lnγ] = collect(np+1:np+nrr);
    np += nrr
    indices[:u] = idx_u
    indices[:m] = idx_bold
    indices[:sts] = idx_sts

    # continue with prior variances
    paramvariance = copy(parammean)
    paramvariance[:lnγ] = ones(Float64, nrr)./64.0;
    paramvariance[:lnα] = ones(Float64, length(parammean[:lnα]))./64.0;
    paramvariance[:lnβ] = ones(Float64, length(parammean[:lnβ]))./64.0;
    for (k, v) in paramvariance
        if occursin("A", string(k))
            paramvariance[k] = ones(length(v))./64.0;
        elseif occursin("κ", string(k))
            paramvariance[k] = ones(length(v))./256.0;
        elseif occursin("ϵ", string(k))
            paramvariance[k] = 1/256.0;
        elseif occursin("τ", string(k))
            paramvariance[k] = 1/256.0;
        end
    end
    return parammean, paramvariance, indices
end

"""
    function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, indices)

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
    - `indices`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
"""
function setup_sDCM(data, model, initcond, csdsetup, priors, hyperpriors, indices, modelparam, modality)
    # compute cross-spectral density
    dt = csdsetup.dt;                      # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freq = csdsetup.freq;                  # frequencies at which the CSD is evaluated
    mar_order = csdsetup.mar_order;        # order of MAR
    _, vars = get_eqidx_tagged_vars(model, "measurement")
    data = Matrix(data[:, String.(Symbol.(vars))])           # make sure the column order is consistent with the ordering of variables of the model that represent the measurements
    mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freq, dt^-1);     # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    statevals = [v for v in values(initcond)]
    append!(statevals, zeros(length(unknowns(model)) - length(statevals)))
    f_model = generate_function(model; expression=Val{false})[1]
    f_at(params, t) = states -> f_model(states, MTKParameters(model, params)..., t)
    derivatives = par -> jacobian(f_at(addnontunableparams(par, model), t), statevals)

    μθ_pr = vecparam(priors.μθ_pr)        # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = diagm(vecparam(priors.Σθ_pr))

    ### Collect prior means and covariances ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = csd_Q(y_csd);             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)
    nr = length(indices[:lnγ])        # region specific noise parameter can be used to get the number of regions

    f = params -> csd_mtf(freq, mar_order, derivatives, params, indices, modality)

    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLState(
        0,                                   # iter
        -4,                                  # log ascent rate
        [-Inf],                              # free energy
        Float64[],                           # delta free energy
        hyperpriors.μλ_pr,                   # metaparameter, initial condition. TODO: why are we not just using the prior mean?
        zeros(np),                           # parameter estimation error ϵ_θ
        [zeros(np), hyperpriors.μλ_pr],      # memorize reset state
        μθ_pr,                               # parameter posterior mean
        Σθ_pr,                               # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLSetup(
        f,                                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                                # empirical cross-spectral density
        1e-1,                                 # tolerance
        [nr, np, ny, nq, nh],                 # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, hyperpriors.μλ_pr],           # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors.Πλ_pr],      # parameter and hyperparameter prior precision matrices
        Q,                                    # components of data precision matrix
        modelparam
    )
    return (vlstate, vlsetup)
end

function run_sDCM_iteration!(state::VLState, setup::VLSetup)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (nr, np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)

    norm_dfdp = opnorm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);

            # E-Step: update
            ϵ_θ += integration_step(dFdθθ, dFdθ, v)

            μθ_po = μθ_pr + ϵ_θ

            dfdp = ForwardDiff.jacobian(f, μθ_po)

            # check for stability
            norm_dfdp = opnorm(dfdp, Inf);
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
    dFdλ = zeros(real(eltype(J)), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        if nh > 1
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
        else
            # if nh == 1, do the followng simplifications to improve computational speed:          
            # 1. replace trace(PΣ[1]) * nq by ny
            # 2. replace JPJ[1] by Pp
            dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;

            # 3. replace trace(PΣ[1],PΣ[1]) * nq by ny
            dFdλλ[1, 1] = - ny/2;
        end

        dFdλλ = dFdλλ + diagm(dFdλ);      # add second order terms; noting diΣ/dλ(i)dλ(i) = diΣ/dλ(i) = P{i}

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        # E-Step: update
        dλ = real(integration_step(dFdλλ, dFdλ, 4))

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
    dθ = integration_step(dFdθθ, dFdθ, v, true)

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
