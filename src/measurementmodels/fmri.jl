"""
fmri.jl 

Models of fMRI signals.

hemodynamics : computes hemodynamic responses and its Jacobian
boldsignal   : computes BOLD signal and gradient
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
mutable struct Hemodynamics{T} <: NBComponent
    p_dict::T
    connector::Num
    odesystem::ODESystem
    function Hemodynamics(;name, lnκ=0.0, lnτ=0.0)
        #= hemodynamic parameters
            H(1) - signal decay                                   d(ds/dt)/ds)
            H(2) - autoregulation                                 d(ds/dt)/df)
            H(3) - transit time                                   (t0)
            H(4) - exponent for Fout(v)                           (alpha)
            H(5) - resting state oxygen extraction                (E0)
        =#
        H = [0.64, 0.32, 2.00, 0.32, 0.4]
        para_dict = scope_dict!(Dict(:lnκ => lnκ, :lnτ => lnτ))
        lnκ=para_dict[:lnκ]
        lnτ=para_dict[:lnτ]
        states = @variables s(t) lnf(t) lnν(t) lnq(t) jcn(t)

        eqs = [
            D(s)   ~ jcn - H[1]*exp(lnκ)*s - H[2]*(exp(lnf) - 1),
            D(lnf) ~ s / exp(lnf),
            D(lnν) ~ (exp(lnf) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν)),
            D(lnq) ~ (exp(lnf)/exp(lnq)*((1 - (1 - H[5])^(exp(lnf)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
        ]
        odesys = ODESystem(eqs, t, states, values(para_dict); name=name)
        new(para_dict, Num(0), odesys)
    end
end


mutable struct LinHemo <: NBComponent
    connector::Num
    bloxinput::Num
    odesystem::ODESystem
    function LinHemo(;name, lnκ=0.0, lnτ=0.0)
        @variables jcn(t)
        @named nmm = LinearNeuralMassBlox()
        @named hemo = Hemodynamics(;lnκ=lnκ, lnτ=lnτ)

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => nmm, :jcn => jcn))
        add_vertex!(g, :blox, hemo)
        add_edge!(g, 1, 2, :weight, 1.0)
        linhemo = ODEfromGraph(g; name=name)
        new(linhemo.nmm₊x, linhemo.jcn, linhemo)
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
    vars = @variables bold(t) lnq(t) lnν(t)

    eqs = [
        bold ~ V0*(k1 - k1*exp(lnq) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
    ]

    ODESystem(eqs, t, vars, params; name=name)
end