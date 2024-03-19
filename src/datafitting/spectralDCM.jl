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
    λ, V = F.values, F.vectors
    local ∂λ_agg, ∂V_agg
    # compute eigenvalue and eigenvector derivatives for all partials
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)   # why do only copy when complex??
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
    ∂V = Array{Partials}(undef, nd, nd)
    ∂λ = Array{Partials}(undef, nd)
    # reassemble the aggregated vectors and values into a Partials type
    for i = 1:nd
        ∂λ[i] = Partials(Tuple(∂λ_agg[i, :]))
        for j = 1:nd
            ∂V[i, j] = Partials(Tuple(∂V_agg[i, j, :]))
        end
    end
    if eltype(V) <: Complex
        evals = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.values, ∂λ)
        evecs = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.vectors, ∂V)
    else
        evals = Dual{T, Float64, length(∂λ[1])}.(F.values, ∂λ)
        evecs = Dual{T, Float64, length(∂V[1])}.(F.vectors, ∂V)
    end
    return Eigen(evals, evecs)
end

function transferfunction_fmri(ω, derivatives, params, params_idx)
    ∂f = derivatives[:∂f](params[params_idx[:evolpars]])
    if ∂f isa Vector
        ∂f = reshape(∂f, sqrt(length(∂f)), sqrt(length(∂f)))
    end

    dfdu = [0.0625  0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0625  0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0625
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0
    0.0     0.0     0.0]

    F = eigen(∂f)
    Λ = F.values
    V = F.vectors

    ∂g = derivatives[:∂g](params[params_idx[:obspars]][1])
    dgdv = ∂g*V
    dvdu = V\dfdu          # u is external variable which we don't use right now. With external variable this would read V/dfdu

    nω = size(ω, 1)            # number of frequencies
    ng = size(∂g, 1)           # number of outputs
    nu = size(dvdu, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(dvdu))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*ω .- Λ[k]).^-1
                S[:,i,j] .+= dgdv[i,k]*dvdu[k,j]*Sk
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
        Gu[:, i, i] .+= exp(α[1])*G
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
        return maximum(vec(sum(abs.(M),dims=1)))
    elseif p == Inf
        return maximum(vec(sum(abs.(M),dims=2)))
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
    Q = inv(Q .+ matlab_norm(Q, 1)/32*Matrix(I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
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
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
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
        if (typeof(v) <: Array)
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
    variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)

    Computes parameter estimation using variational Laplace that is to a large extend equivalent to the SPM12 implementation
    and provides the exact same values.

    Arguments:
    - `idx_A`: indices of connection weight parameter matrix A in model Jacobian
    - `y`: empirical cross-spectral density (input data)
    - `derivatives`: jacobian of model as well as gradient of observer function
    - `w`: fequencies at which to estimate cross-spectral densities
    - `V`: projection matrix from full parameter space to reduced space that removes parameters with zero variance prior
    - `p`: order of multivariate autoregressive model for estimation of cross-spectral densities from data
    - `priors`: Bayesian priors, mean and variance thereof. Laplace approximation assumes Gaussian distributions
    - `niter`: number of iterations of the optimization procedure
"""
@views function variationalbayes(y, derivatives, w, V, p, priors, niter)
    # extract priors
    Πθ_pr = priors[:Σ][:Πθ_pr]
    Πλ_pr = priors[:Σ][:Πλ_pr]
    μλ_pr = priors[:Σ][:μλ_pr]
    Q = priors[:Σ][:Q]

    # prep stuff
    μθ_pr = vecparam(priors[:μ])      # note: μθ_po is posterior and μθ_pr is prior
    nd = size(y, 2)
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    nq = 1
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)
    ϵ_θ = zeros(np)  # M.P - μθ_pr # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ with μθ_pr above.
    μθ_po = μθ_pr + V*ϵ_θ

    revert = false
    f_prep = params -> csd_fmri_mtf(w, p, derivatives, params, params_idx)

    # state variable
    F = -Inf
    F0 = F
    previous_F = F
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    state = vb_state(0, F, λ, zeros(np), μθ_po, inv(Πθ_pr))
    dfdp = zeros(ComplexF64, length(w)*nd^2, np)
    local ϵ_λ, iΣ, Σλ, Σθ, dFdpp, dFdp
    for k = 1:niter
        state.iter = k
        dfdp = jacobian(f_prep, μθ_po) * V
        norm_dfdp = matlab_norm(dfdp, Inf);
        revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

        if revert && k > 1
            for i = 1:4
                # reset expansion point and increase regularization
                v = min(v - 2,-4);
                t = exp(v - logdet(dFdpp)/np)

                # E-Step: update
                if t > exp(16)
                    ϵ_θ = state.ϵ_θ - dFdpp \ dFdp    # -inv(dfdx)*f
                else
                    ϵ_θ = state.ϵ_θ + expv(t, dFdpp, dFdpp \ dFdp) - dFdpp \ dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
                end

                μθ_po = μθ_pr + V*ϵ_θ

                # J_test = JacVec(f_prep, μθ_po)
                # dfdp = stack(J_test*v for v in eachcol(V))
                dfdp = jacobian(f_prep, μθ_po) * V

                # check for stability
                norm_dfdp = matlab_norm(dfdp, Inf);
                revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

                # break
                if ~revert
                    break
                end
            end
        end

        f = f_prep(μθ_po)
        ϵ = reshape(y - f, ny)                   # error value
        J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

        ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
        P = zeros(eltype(J), size(Q))
        PΣ = zeros(eltype(J), size(Q))
        JPJ = zeros(real(eltype(J)), size(J,2), size(J,2), size(Q,3))
        dFdλ = zeros(eltype(J), nh)
        dFdλλ = zeros(real(eltype(J)), nh, nh)
        for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
            iΣ = zeros(eltype(J), ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end

            Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why?
            Σθ = inv(Pp + Πθ_pr)

            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = iΣ \ P[:,:,i]
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above), what's the rational?
            end
            for i = 1:nh
                dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                    dFdλλ[j, i] = dFdλλ[i, j]
                end
            end

            ϵ_λ = λ - μλ_pr
            dFdλ = dFdλ - Πλ_pr*ϵ_λ
            dFdλλ = dFdλλ - Πλ_pr
            Σλ = inv(-dFdλλ)

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
        L[2] = (logdet(Πθ_pr * Σθ) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
        L[3] = (logdet(Πλ_pr * Σλ) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
        F = sum(L)

        if k == 1
            F0 = F
        end

        if F > state.F || k < 3
            # accept current state
            state.ϵ_θ = ϵ_θ
            state.λ = λ
            state.Σθ = Σθ
            state.μθ_po = μθ_po
            state.F = F
            # Conditional update of gradients and curvature
            dFdp  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
            dFdpp = -real(J' * iΣ * J) - Πθ_pr
            # decrease regularization
            v = min(v + 1/2, 4);
        else
            # reset expansion point
            ϵ_θ = state.ϵ_θ
            λ = state.λ
            # and increase regularization
            v = min(v - 2, -4);
        end

        # E-Step: update
        t = exp(v - spm_logdet(dFdpp)/np)
        if t > exp(16)
            dθ = - inv(dFdpp) * dFdp    # -inv(dfdx)*f
        else
            dθ = exponential!(t * dFdpp) * inv(dFdpp) * dFdp - inv(dFdpp) * dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
        end

        ϵ_θ += dθ
        μθ_po = μθ_pr + V*ϵ_θ
        dF = dot(dFdp, dθ);

        # convergence condition: reach a change in Free Energy that is smaller than 0.1 four consecutive times
        print("iteration: ", k, " - F:", state.F - F0, " - dF predicted:", dF, "\n")
        criterion = vcat(dF < 1e-1, criterion[1:end - 1]);
        if all(criterion)
            print("convergence\n")
            break
        end
    end
    print("iterations terminated\n")
    state.F = F
    state.Σθ = V*Σθ*V'
    state.μθ_po = μθ_po
    return state
end

"""
    spectralVI(data, neuraldynmodel, observationmodel, initcond, csdsetup, params, hyperparams)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

    Arguments:
    - `data`             : Dataframe with column names corresponding to the regions of measurement.
    - `neuraldynmodel`   : MTK model, it is an ODESystem or a System (haven't tested with System yet).
    - `observationmodel` : MTK model that defines measurement function (ex. bold signal). 
                           Current implementation limits to one measurement functional form for all regions.
    - `initcond`         : Dictionary of initial conditions, numerical values for all states
    - `csdsetup`         : Dictionary of parameters required for the computation of the cross spectral density
    -- `dt`              : sampling interval
    -- `freq`            : frequencies at which to evaluate the CSD
    -- `p`               : order parameter of the multivariate autoregression model
    - `params`           : Dataframe of parameters with the following columns:
    -- `name`            : corresponds to MTK model name
    -- `mean`            : corresponds to prior mean value
    -- `variance`        : corresponds to the prior variances
    - `hyperparams`      : Dataframe of parameters with the following columns:
    -- `Πλ_pr`           : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`           : prior mean(s) for λ hyperparameter(s)
"""
function spectralVI(data, neuraldynmodel, observationmodel, initcond, csdsetup, priors, hyperpriors)
    # compute cross-spectral density
    y = Matrix(data)
    nr = ncol(data)                     # number of regions
    ns = length(initcond)               # number of states in total
    dt = csdsetup[:dt]                  # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freqs = csdsetup[:freq]             # frequencies at which the CSD is evaluated
    p = csdsetup[:p]                    # order of MAR
    mar = mar_ml(y, p)                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freqs, dt^-1)  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    grad_full = function(grad, obsstates, obsidx, params, nr, ns)
        tmp = zeros(typeof(params), nr, ns)
        for i in 1:nr
            tmp[i, obsidx[i]] = grad(vcat([1], obsstates[i]), params, t)[2:end] # [(length(obsstates[i])+1):-1:2] TODO: using the reverse will improve results but is wrong.
        end
        return tmp
    end

    jac_f = generate_jacobian(neuraldynmodel, expression = Val{false})[1]
    grad_g = generate_jacobian(observationmodel, expression = Val{false})[1]   # computes gradient since output is a scalar

    statevals = [v for v in values(initcond)]
    obs = get_hemodynamic_observers(neuraldynmodel, nr)
    obsstates = map(obs -> [initcond[s] for s in obs], values(obs[2]))
    derivatives = Dict(:∂f => par -> jac_f(statevals, addnontunableparams(par, neuraldynmodel), t),
                       :∂g => par -> grad_full(grad_g, obsstates, obs[1], par, nr, ns))

    θΣ = diagm(vecparam(OrderedDict(priors.name .=> priors.variance)))
    # depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
    # Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
    # to what extend this is a the most reasonable approach. 
    idx = findall(x -> x != 0, θΣ);
    V = zeros(size(θΣ, 1), length(idx));
    order = sortperm(θΣ[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0

    ### Collect prior means and covariances ###
    Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
    priors = Dict(:μ => OrderedDict(priors.name .=> priors.mean),
                  :Σ => Dict(
                            :Πθ_pr => inv(θΣ),               # prior model parameter precision
                            :Πλ_pr => hyperpriors[:Πλ_pr],   # prior metaparameter precision
                            :μλ_pr => hyperpriors[:μλ_pr],   # prior metaparameter mean
                            :Q => Q                          # decomposition of model parameter covariance
                            )
                  );

    ### Compute the variational Bayes with Laplace approximation ###
    return variationalbayes(y_csd, derivatives, freqs, V, p, priors, 128)
end

function setup_sDCM(data, stateevolutionmodel, observationmodel, initcond, csdsetup, priors, hyperpriors, params_idx)
    # compute cross-spectral density
    y = Matrix(data);
    nr = ncol(data);                     # number of regions
    sts = states(stateevolutionmodel)    # variables of model
    ns = length(sts)                     # number of states in total

    dt = csdsetup[:dt];                  # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    ω = csdsetup[:freq];                 # frequencies at which the CSD is evaluated
    p = csdsetup[:p];                    # order of MAR
    mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, ω, dt^-1);      # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    grad_full = function(grad, obsstates, obsidx, params, nr, ns)
        tmp = zeros(typeof(params), nr, ns)
        for i in 1:nr
            tmp[i, obsidx[i]] = grad(vcat([1], obsstates[i]), params, t)[2:end] # [(length(obsstates[i])+1):-1:2] TODO: using the reverse will improve results but is wrong.
        end
        return tmp
    end

    jac_f = generate_jacobian(stateevolutionmodel, expression = Val{false})[1]
    grad_g = generate_jacobian(observationmodel, expression = Val{false})[1]     # computes gradient since output is a scalar

    statevals = [v for v in values(initcond)]
    # match states of observation model with different states of evolution model
    obs = get_hemodynamic_observers(stateevolutionmodel, nr)
    obsstates = Dict(map((v, k) -> k => [initcond[s] for s in v], values(obs[2]), keys(obs[2])))
    derivatives = Dict(:∂f => par -> jac_f(statevals, addnontunableparams(par, stateevolutionmodel), t),
                       :∂g => par -> grad_full(grad_g, obsstates, obs[1], par, nr, ns))

    μθ_pr = vecparam(OrderedDict(priors.name .=> priors.mean))            # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = diagm(vecparam(OrderedDict(priors.name .=> priors.variance)))

    ### Collect prior means and covariances ###
    Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)

    f = params -> csd_fmri_mtf(ω, p, derivatives, params, params_idx)

    # dfdp = zeros(ComplexF64, length(ω)*nd^2, np)
    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLState(
        0,             # iter
        -4,            # log ascent rate
        [-Inf],        # free energy
        [],            # delta free energy
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
        [μθ_pr, hyperpriors[:μλ_pr]],            # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors[:Πλ_pr]],     # parameter and hyperparameter prior precision matrices
        Q                                   # components of data precision matrix
    )
    return (vlstate, vlsetup)
end

function run_sDCM_iteration!(state::VLState, setup::VLSetup)
    μθ_po = state.μθ_po

    λ = state.λ
    v = state.v
    ϵ_θ = state.ϵ_θ
    dFdθ = state.dFdθ
    dFdθθ = state.dFdθθ

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    # Πθ_pr = deserialize("tmp.dat")[vcat(1:20, 24), :]' *Πθ_pr* deserialize("tmp.dat")[vcat(1:20, 24), :]
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)# * deserialize("tmp.dat")[vcat(1:20, 24), :]

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

            dfdp = jacobian(f, μθ_po) #* deserialize("tmp.dat")[vcat(1:20, 24), :]

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