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

function transferfunction_fmri(w, sts, derivatives, params)   # relates to: spm_dcm_mtf.m

    C = params[:C]
    C /= 16.0   # TODO: unclear why C is devided by 16 but see spm_fx_fmri.m:49

    # 2. get jacobian of hemodynamics
    ∂f = substitute(derivatives[:∂f], params)
    ∂f = convert(Array{Real}, substitute(∂f, sts))
    idx_A = findall(occursin.("A[", string.(derivatives[:∂f])))
    A = ∂f[idx_A]
    nd = Int(sqrt(length(A)))
    A_tmp = A[[(i-1)*nd+i for i=1:nd]]
    A[[(i-1)*nd+i for i=1:nd]] -= exp.(A_tmp)/2 + A_tmp
    ∂f[idx_A] = A
    # if I eventually need also the change of variables rather than just the derivative then here is where to fix it! 
    dfdu = [diagm(C);
            zeros(size(∂f, 1)-nd, length(C))]

    F = eigen(Symbolics.value.(∂f), sortby=nothing, permute=true)
    Λ = F.values
    V = F.vectors

    ∂g = substitute(derivatives[:∂g], params)
    ∂g = Symbolics.value.(substitute(∂g, sts))
    dgdv = ∂g*V
    dvdu = pinv(V)*dfdu

    nw = size(w,1)            # number of frequencies
    ng = size(∂g,1)           # number of outputs
    nu = size(dfdu,2)         # number of inputs
    nk = size(V,2)            # number of modes
    S = zeros(Complex, nw, ng, nu)

    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*w .- Λ[k]).^-1    # TODO: clean up 1im*2*pi*freq instead of omega to be consistent with the usual nomenclature
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
function csd_approx(w, sts, derivatives, param)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nw = length(w)
    nd = size(x, 1)
    α = param[:lnα]
    β = param[:lnβ]
    γ = param[:lnγ]
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    Gu = zeros(nw, nd, nd)
    Gn = zeros(nw, nd, nd)
    G = w.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
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
    S = transferfunction_fmri(w, sts, derivatives, param)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(ComplexF64,nw,nd,nd);
    for i = 1:nw
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end

function csd_fmri_mtf(freqs, p, sts, derivatives, param)   # alongside the above realtes to spm_csd_fmri_mtf.m
    G = csd_approx(freqs, sts, derivatives, param)
    dt = 1/(2*freqs[end])
    # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
    # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
    # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. Friston conferms that likely it is
    # to make y well behaved.
    mar = csd2mar(G, freqs, dt, p-1)
    y = mar2csd(mar, freqs)
    return y
end

function diff(U, dx, f, param::OrderedDict)
    nJ = size(U, 2)
    y0 = f(param)
    J = zeros(ComplexF64, nJ, size(y0, 1), size(y0, 2), size(y0, 3))
    for i = 1:nJ
        tmp_param = vecparam(param) .+ U[:, i]*dx
        y1 = f(unvecparam(tmp_param, param))
        J[i,:,:,:] = (y1 .- y0)/dx
    end
    return J, y0
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

function vecparam(param::OrderedDict{Any,Any})
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


function variationalbayes(sts, y, derivatives, w, V, p, priors, niter)    # relates to spm_nlsi_GN.m
    # extract priors
    Πθ_pr = priors[:Σ][:Πθ_pr]
    Πλ_pr = priors[:Σ][:Πλ_pr]
    μλ_pr = priors[:Σ][:μλ_pr]
    Q = priors[:Σ][:Q]
    μθ_pr = vecparam(priors[:μ])            # note: μθ_po is posterior and θμ is prior

    # prep stuff
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    nq = 1
    nh = size(Q, 3)            # number of precision components/hyper parameters
    λ = 8*ones(nh)
    ϵ_θ = zeros(np)  # M.P - θμ # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ_po with θμ above.
    μθ_po = μθ_pr + V*ϵ_θ

    dx = exp(-8)
    revert = false
    f_prep = pars -> csd_fmri_mtf(w, p, sts, derivatives, pars)

    # state variable
    F = -Inf
    F0 = F
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    state = vb_state(0, F, λ, zeros(np), μθ_po, inv(Πθ_pr))
    local ϵ_λ, iΣ, Σλ, Σθ, dFdθθ, dFdθ
    dFdλ = zeros(ComplexF64, nh)
    dFdλλ = zeros(Float64, nh, nh)
    for k = 1:niter
        state.iter = k

        dfdθ, f = diff(V, dx, f_prep, unvecparam(μθ_po, priors[:μ]));
        dfdθ = transpose(reshape(dfdθ, np, ny))
        norm_dfdθ = matlab_norm(dfdθ, Inf);      # NB that the norm in Julia is different from MATLAB. For consistency with SPM12 we reimplemented it here
        revert = isnan(norm_dfdθ) || norm_dfdθ > exp(32);

        if revert && k > 1
            for i = 1:4
                # reset expansion point and increase regularization
                v = min(v - 2, -4);
                t = exp(v - logdet(dFdθθ)/np)

                # E-Step: update
                if t > exp(16)
                    ϵ_θ = state.ϵ_θ - dFdθθ\dFdθ    # -inv(dfdx)*f
                else
                    idFdθθ = inv(dFdθθ)
                    ϵ_θ = state.ϵ_θ + expv(t, dFdθθ, idFdθθ*dFdθ) - idFdθθ*dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
                end

                μθ_po = μθ_pr + V*ϵ_θ

                dfdθ, f = diff(V, dx, f_prep, unvecparam(μθ_po, priors[:μ]));
                dfdθ = transpose(reshape(dfdθ, np, ny))

                # check for stability
                norm_dfdθ = matlab_norm(dfdθ, Inf);
                revert = isnan(norm_dfdθ) || norm_dfdθ > exp(32);

                # break
                if ~revert
                    break
                end
            end
        end


        ϵ = reshape(y - f, ny)    # error value
        J = - dfdθ   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 


        ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
        for m = 1:8   # 8 seems arbitrary. This is probably because optimization falls often into a periodic orbit. ToDo: Issue #8
            iΣ = zeros(ComplexF64, ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end
            Σ = inv(iΣ)               # Julia requires conversion to dense matrix before inversion so just use dense to begin with
            Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why?
            Σθ = inv(Pp + Πθ_pr)

            P = similar(Q)
            PΣ = similar(Q)
            JPJ = zeros(size(Pp,1), size(Pp,2), size(Q,3))
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = P[:,:,i] * Σ
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above), what's the rational?
            end

            for i = 1:nh
                dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ,P[:,:,i],ϵ)) - tr(Σθ * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdλλ[i, j] = - real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2     # eps = randn(sizen), (eps' * Ai) * (Aj * eps)
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
                dλ = -real(inv(dFdλλ) * dFdλ)
            else
                idFdλλ = inv(dFdλλ)
                dλ = real(expv(t, dFdλλ, idFdλλ*dFdλ) - idFdλλ*dFdλ)   # (expm(dfdx*t) - I)*inv(dfdx)*f
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
        L = zeros(3)
        L[1] = (real(logdet(iΣ))*nq  - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
        L[2] = (logdet(Πθ_pr * Σθ) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
        L[3] = (logdet(Πλ_pr * Σλ) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2;
        F = sum(L);

        if k == 1
            F0 = F
        end

        if F > state.F || k < 3
            # accept current state
            state.F = F
            state.ϵ_θ = ϵ_θ
            state.λ = λ
            state.Σθ = Σθ
            state.μθ_po = μθ_po
            # Conditional update of gradients and curvature
            dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ
            dFdθθ = -real(J' * iΣ * J) - Πθ_pr
            # decrease regularization
            v = min(v + 1/2,4);
        else
            # reset expansion point
            ϵ_θ = state.ϵ_θ
            λ = state.λ
            # and increase regularization
            v = min(v - 2,-4);
        end

        # E-Step: update
        t = exp(v - spm_logdet(dFdθθ)/np)
        if t > exp(16)
            dθ = -inv(dFdθθ)*dFdθ    # -inv(dfdx)*f
        else
            dθ = exp(t * dFdθθ) * inv(dFdθθ)*dFdθ - inv(dFdθθ)*dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
        end

        ϵ_θ += dθ
        μθ_po = μθ_pr + V*ϵ_θ
        dF  = dot(dFdθ, dθ);

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
