"""
fmri.jl 

Models of fMRI signals.

hemodynamics! : computes hemodynamic responses and its Jacobian
boldsignal    : computes BOLD signal and gradient
"""



"""
This function implements the hymodynamics model (balloon model and neurovascular state eq.) described in:     
Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.

adapted from spm_fx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan

### Input variables ###
na         : neural activity of dimension N
x          : biophysical quantities of hemodynamic response to neural activity of dimension Nx4, components are:
    x[:,1] : vascular signal: s
    x[:,2] : rCBF: ln(f)
    x[:,3] : venous volume: ln(ν)
    x[:,4] : deoxyhemoglobin (dHb): ln(q)
decay      : exponential prefactor to signal decay H[1], set to 0 for standard parameter value.
transit    : exponential prefactor to transit time H[3], set to 0 for standard parameter value.
NB: both transit time and decay can change for each neural signal dimension N.

### Return variables ###
dx         : left hand side of differential equation
J          : analytic jacobian of hemodynamic response
"""
function hemodynamics!(dx, x, na, lndecay, lntransit)
    #= hemodynamic parameters
        H[1] : signal decay                                   d(ds/dt)/ds)
        H[2] : autoregulation                                 d(ds/dt)/df)
        H[3] : transit time                                   (t0)
        H[4] : exponent for Fout(v)                           (alpha)
        H[5] : resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    # exponentiation of hemodynamic state variables
    x[:, 2:4] = exp.(x[:, 2:4])

    # signal decay
    κ = H[1]*exp(lndecay)

    # transit time
    τ = H[3]*exp.(lntransit)

    # Fout = f(v) - outflow
    fv = x[:, 3].^(H[4]^-1)

    # e = f(f) - oxygen extraction
    ff = (1.0 .- (1.0 - H[5]).^(x[:, 2].^-1))/H[5]

    # implement differential state equation f = dx/dt (hemodynamic)
    dx[:, 1] = na .- κ.*x[:, 1] .- H[2]*(x[:, 2] .- 1)   # Corresponds to eq (9)
    dx[:, 2] = x[:, 1]./x[:, 2]      # Corresponds to eq (10), note the added logarithm (see doc string)
    dx[:, 3] = (x[:, 2] .- fv)./(τ.*x[:, 3])    # Corresponds to eq (8), note the added logarithm (see doc string)
    dx[:, 4] = (ff.*x[:, 2] .- fv.*x[:, 4]./x[:, 3])./(τ.*x[:, 4])  # Corresponds to eq (8), note the added logarithm (see doc string)

    N = size(x, 1)        # number of dimensions, equals typically number of regions
    J = zeros(4N, 4N)

    J[1:N,1:N] = la.Matrix(-κ*la.I, N, N)   # TODO: make it work when κ/decay is a vector. Only solution if-clause? la.diagm doesn't allow scalars, [κ] would work in that case
    J[1:N,N+1:2N] = la.diagm(-H[2]*x[:,2])
    J[N+1:2N,1:N] = la.diagm( x[:,2].^-1)
    J[N+1:2N,N+1:2N] = la.diagm(-x[:,1]./x[:,2])
    J[2N+1:3N,N+1:2N] = la.diagm(x[:,2]./(τ.*x[:,3]))
    J[2N+1:3N,2N+1:3N] = la.diagm(-x[:,3].^(H[4]^-1 - 1)./(τ*H[4]) - (x[:,3].^-1 .*(x[:,2] - x[:,3].^(H[4]^-1)))./τ)
    J[3N+1:4N,N+1:2N] = la.diagm((x[:,2] .+ log(1 - H[5])*(1 - H[5]).^(x[:,2].^-1) .- x[:,2].*(1 - H[5]).^(x[:,2].^-1))./(τ.*x[:,4]*H[5]))
    J[3N+1:4N,2N+1:3N] = la.diagm((x[:,3].^(H[4]^-1 - 1)*(H[4] - 1))./(τ*H[4]))
    J[3N+1:4N,3N+1:4N] = la.diagm((x[:,2]./x[:,4]).*((1 - H[5]).^(x[:,2].^-1) .- 1)./(τ*H[5]))

    return (dx, J)
end


"""
This function implements the BOLD signal model described in: 

Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.

adapted from spm_gx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan

### Input variables ###
lnν  : venous volume: ln(ν)
lnq  : deoxyhemoglobin (dHb): ln(q)
lnϵ  : ratio of intra- to extra-vascular components (note also here as above, actually ln(ϵ) rather than ϵ itself)

### Return variables ###
bold : BOLD response (%)
∇    : analytic gradient of bold signal function
"""
function boldsignal(lnν, lnq, lnϵ)

    #=
        NB: Biophysical constants for 1.5T scanners:
        TE  = 0.04
        V0  = 4    
        r0  = 25
        nu0 = 40.3
        E0  = 0.4
    =#

    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE = 0.04
    # resting venous volume (%)
    V0 = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0 = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0 = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
    ϵ = exp(lnϵ)

    # Coefficients in BOLD signal model
    k1 = 4.3*nu0*E0*TE;
    k2 = ϵ*r0*E0*TE;
    k3 = 1 - ϵ;
    # Output equation of BOLD signal model
    ν = exp.(lnν)
    q = exp.(lnq)
    bold = V0*(k1 .- k1*q .+ k2 .- k2*q./ν .+ k3 .- k3*ν)

    nd = length(lnν)
    ∇ = zeros(nd, 2nd)
    ∇[1:nd, 1:nd]     = la.diagm(-V0*(k3*ν .- k2*q./ν))    # TODO: it is unclear why this is the correct gradient, do the algebra... (note this is a gradient per area, not a Jacobian, BOLD signal is a scalar function)
    ∇[1:nd, nd+1:2nd] = la.diagm(-V0*(k1*q .+ k2*q./ν))

    return (bold, ∇)
end