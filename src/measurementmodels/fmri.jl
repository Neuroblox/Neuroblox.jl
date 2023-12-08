"""
fmri.jl 

Models of fMRI signals.

BalloonModel : computes hemodynamic responses
boldsignal   : computes BOLD signal
"""

"""
### Input variables ###
name : name of ODE system
jcn  : neural activity
s    : vascular signal
lnf  : logarithm of rCBF
lnν  : logarithm of venous volume
lnq  : logarithm of deoxyhemoglobin (dHb)

### Parameter ###
lnκ  : logarithmic prefactor to signal decay H[1], set to 0 for standard parameter value.
lnτ  : logarithmic prefactor to transit time H[3], set to 0 for standard parameter value.

### Return variables ###
returns an ODESystem of the biophysical model for the hemodynamics
"""
struct BalloonModel <: ObserverBlox
    params
    output
    jcn
    odesystem
    namespace
    function BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0)
        #= hemodynamic parameters
            H(1) - signal decay                                   d(ds/dt)/ds)
            H(2) - autoregulation                                 d(ds/dt)/df)
            H(3) - transit time                                   (t0)
            H(4) - exponent for Fout(v)                           (alpha)
            H(5) - resting state oxygen extraction                (E0)
        =#

        H = [0.64, 0.32, 2.00, 0.32, 0.4]
        p = progress_scope(lnκ, lnτ)  # progress scope if needed
        p = compileparameterlist(lnκ=p[1], lnτ=p[2])  # finally compile all parameters
        lnκ, lnτ = p  # assign the modified parameters
        
        sts = @variables s(t)=1.0 lnf(t)=1.0 lnν(t)=1.0 [output=true, description="hemodynamic_observer"] lnq(t)=1.0 [output=true, description="hemodynamic_observer"] jcn(t)=0.0 [input=true]

        eqs = [
            D(s)   ~ jcn - H[1]*exp(lnκ)*s - H[2]*(exp(lnf) - 1),
            D(lnf) ~ s / exp(lnf),
            D(lnν) ~ (exp(lnf) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν)),
            D(lnq) ~ (exp(lnf)/exp(lnq)*((1 - (1 - H[5])^(exp(lnf)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
        ]
        sys = System(eqs, name=name)
        new(p, Num(0), sts[5], sys, namespace)
    end
end



"""
BOLD signal model as described in: 

Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.

### Input variables ###
lnν  : logarithm of venous volume
lnq  : logarithm of deoxyhemoglobin (dHb)

### Parameter ###
lnϵ  : logarithm of ratio of intra- to extra-vascular signal

### Return variables ###
returns ODESystem
"""
function boldsignal(;name, lnϵ=0.0)
    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE

    params = @parameters lnϵ=lnϵ
    vars = @variables bold(t) lnν(t) lnq(t)   # TODO: got to be really careful with the current implementation! A simple permutation of this breaks the algorithm!

    eqs = [
        bold ~ V0*(k1 - k1*exp(lnq) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
    ]

    ODESystem(eqs, t, vars, params; name=name)
end