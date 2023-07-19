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

using ForwardDiff
using ForwardDiff: Dual
using ForwardDiff: Partials
using FFTW: ifft
using LinearAlgebra: Eigen
using LinearAlgebra
using RuntimeGeneratedFunctions
# using ToeplitzMatrices
# using ExponentialUtilities

ForwardDiff.can_dual(::Type{Complex{Float64}}) = true
using ChainRules: _eigen_norm_phase_fwd!
tagtype(::Dual{T,V,N}) where {T,V,N} = T


function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{eltype(x)}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end

function FFTW.ifft(x::Array{Complex{Dual{T, P, N}}}) where {T, P, N}
    return ifft(real(x)) + 1im*ifft(imag(x))
end

function FFTW.ifft(x::Array{Dual{T, P, N}}) where {T, P, N}
    v = (tmp->tmp.value).(x)
    iftx = ifft(v)
    iftrp = Array{Partials}(undef, length(x))
    iftip = Array{Partials}(undef, length(x))
    local iftrp_agg, iftip_agg
    for i = 1:N
        dx = (tmp->tmp.partials[i]).(x)
        iftdx = ifft(dx)
        if i == 1
            iftrp_agg = real(iftdx) .* dx
            iftip_agg = (1im * imag(iftdx)) .* dx
        else
            iftrp_agg = cat(iftrp_agg, real(iftdx) .* dx, dims=2)
            iftip_agg = cat(iftip_agg, (1im * imag(iftdx)) .* dx, dims=2)
        end
    end
    for i = 1:length(x)
        iftrp[i] = Partials(Tuple(iftrp_agg[i, :]))
        iftip[i] = Partials(Tuple(iftip_agg[i, :]))
    end
    return Complex.(Dual{T, P, N}.(real(iftx), iftrp), Dual{T, P, N}.(imag(iftx), iftip))
end

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


function transferfunction_fmri(w, idx_A, derivatives, params)
    nd = Int(sqrt(length(idx_A)))
    C = params[(6+2nd+nd^2):(5+3nd+nd^2)]

    C /= 16.0   # TODO: unclear why C is devided by 16 but see spm_fx_fmri.m:49
    Main.bar[] = derivatives[:∂f], params
    ∂f = derivatives[:∂f](params[1:(nd^2+nd+1)]) #convert(Array{Real}, substitute(derivatives[:∂f], params))

    dfdu = zeros(eltype(C), size(∂f, 1), length(C))
    dfdu[CartesianIndex.([(idx[2][1], idx[1]) for idx in enumerate(idx_A[[(i-1)*nd+i for i=1:nd]])])] = C

    F = eigen(∂f)
    Λ = F.values
    V = F.vectors

    ∂g = derivatives[:∂g](params[end])
    # ∂g = Symbolics.value.(substitute(∂g, sts))
    dgdv = ∂g*V
    dvdu = V\dfdu

    nw = size(w,1)            # number of frequencies
    ng = size(∂g,1)           # number of outputs
    nu = size(dfdu,2)         # number of inputs
    nk = size(V,2)            # number of modes
    S = zeros(Complex{real(eltype(dvdu))}, nw, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*w .- Λ[k]).^-1
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
function csd_approx(w, idx_A, derivatives, param)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nw = length(w)
    nd = Int(sqrt(length(idx_A)))
    α = param[(2+nd+nd^2):(3+nd+nd^2)]
    β = param[(4+nd+nd^2):(5+nd+nd^2)]
    γ = param[(6+nd+nd^2):(5+2nd+nd^2)]
    
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    G = w.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nw, nd, nd)
    Gn = zeros(eltype(G), nw, nd, nd)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1])*G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = w.^(-exp(β[2])/2)
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
    S = transferfunction_fmri(w, idx_A, derivatives, param)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nw, nd, nd);
    for i = 1:nw
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end

@views function csd_fmri_mtf(freqs, p, idx_A, derivatives, param)   # alongside the above realtes to spm_csd_fmri_mtf.m
    G = csd_approx(freqs, idx_A, derivatives, param)
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


function matlab_norm(A, p)
    if p == 1
        return maximum(vec(sum(abs.(A),dims=1)))
    elseif p == Inf
        return maximum(vec(sum(abs.(A),dims=2)))
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
    Q = inv(Q .+ matlab_norm(Q, 1)/32*la.Matrix(la.I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
end

function spm_logdet(M)
    TOL = 1e-16
    s = la.diag(M)
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = la.svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
end

mutable struct vb_state
    iter::Int
    F::Float64
    λ::Vector{Float64}
    ϵ_θ::Vector{Float64}
    μθ_po::Vector{Float64}
    Σθ::Matrix{Float64}
end

function vecparam(param::OrderedDict{Any, Any})
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

function unvecparam(vals, param::OrderedDict{Any,Any})
    iter = 1
    paramnewvals = copy(param)
    for (k, v) in param
        if (typeof(v) <: Array)
            paramnewvals[k] = vals[iter:iter+length(v)-1]
            iter += length(v)
        else
            paramnewvals[k] = vals[iter]
            iter += 1
        end
    end
    return paramnewvals
end


@views function variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)
    # extract priors
    Πθ_pr = priors[:Σ][:Πθ_pr]
    Πλ_pr = priors[:Σ][:Πλ_pr]
    μλ_pr = priors[:Σ][:μλ_pr]
    Q = priors[:Σ][:Q]

    # prep stuff
    μθ_pr = vecparam(priors[:μ])      # note: μθ_po is posterior and μθ_pr is prior
    nd = size(y,2)
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    nq = 1
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)
    ϵ_θ = zeros(np)  # M.P - μθ_pr # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ with μθ_pr above.
    μθ_po = μθ_pr + V*ϵ_θ
 
    revert = false
    f_prep = param -> csd_fmri_mtf(w, p, idx_A, derivatives, param)

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
        dfdp = ForwardDiff.jacobian(f_prep, μθ_po) * V
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
                dfdp = ForwardDiff.jacobian(f_prep, μθ_po) * V

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
Performs a variational inference to fit a cross spectral density. Current implementation provides a Variational Laplace fit. 
(ToDo: generalize to different VI algorithms)

Input:
- data             : Dataframe with column names corresponding to the regions of measurement.
- neuraldynmodel   : MTK model, it is an ODESystem or a System (haven't tested with System yet).
- observationmodel : MTK model that defines measurement function (ex. bold signal). 
                     Current implementation limits to one measurement functional form for all regions.
- initcond         : Dictionary of initial conditions, numerical values for all states
- csdsetup         : Dictionary of parameters required for the computation of the cross spectral density
-- dt              : sampling interval
-- freq            : frequencies at which to evaluate the CSD
-- p               : order parameter of the multivariate autoregression model
- params           : Dataframe of parameters with the following columns:
-- name            : corresponds to MTK model name
-- mean            : corresponds to prior mean value
-- variance        : corresponds to the prior variances
- hyperparams      : Dataframe of parameters with the following columns:
-- Πλ_pr           : prior precision matrix for λ hyperparameter(s)
-- μλ_pr           : prior mean(s) for λ hyperparameter(s)
"""
function spectralVI(data, neuraldynmodel, observationmodel, initcond, csdsetup, params, hyperparams)
    # compute cross-spectral density
    y = Matrix(data);
    nd = ncol(data);                     # dimension of the data, number of regions
    dt = csdsetup[:dt];                 # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freqs = csdsetup[:freq];            # frequencies at which the CSD is evaluated
    p = csdsetup[:p];                   # order of MAR
    mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    jac_f = calculate_jacobian(neuraldynmodel)
    grad_g = calculate_jacobian(observationmodel)[2:end]

    # define values of states
    all_s = states(neuraldynmodel)

    obs_states = states(observationmodel)
    statesubs = merge.([Dict(obs_states[2] => s) for s in all_s if occursin(string(obs_states[2]), string(s))],
                    [Dict(obs_states[3] => s) for s in all_s if occursin(string(obs_states[3]), string(s))])
    
    # gradient of g for all regions, note that the measurement model g is defined region independent, here it is extended to all regions
    grad_g_full = Num.(zeros(nd, length(all_s)))
    for (i, s) in enumerate(all_s)
        dim = parse(Int64, string(s)[2])
        if occursin.(string(obs_states[2]), string(s))
            grad_g_full[dim, i] = substitute(grad_g[1], statesubs[dim])
        elseif occursin.(string(obs_states[3]), string(s))
            grad_g_full[dim, i] = substitute(grad_g[2], statesubs[dim])
        end
    end

    idx_A = findall(occursin.("A[", string.(jac_f)))
    derivatives = Dict(:∂f => generate_jacobian(neuraldynmodel, expression = Val{true})[1],
                       :∂g => eval(Symbolics.build_function(substitute(grad_g_full, initcond), params.name[end])[1]))
    tmp = derivatives[:∂f]
    Main.foo[] = tmp, initcond, params
    tmp = [v for v in values(initcond)]
    derivatives[:∂f] = p -> derivatives[:∂f](tmp, p, t)
    θΣ = diagm(vecparam(OrderedDict(params.name .=> params.variance)))
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
    priors = Dict(:μ => OrderedDict(params.name .=> params.mean),
    :Σ => Dict(
              :Πθ_pr => inv(θΣ),               # prior model parameter precision
              :Πλ_pr => hyperparams[:Πλ_pr],   # prior metaparameter precision
              :μλ_pr => hyperparams[:μλ_pr],   # prior metaparameter mean
              :Q => Q                          # decomposition of model parameter covariance
              )
            );
    
    ### Compute the variational Bayes with Laplace approximation ###
    return variationalbayes(idx_A, y_csd, derivatives, freqs, V, p, priors, 128)
end