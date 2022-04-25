@parameters t
D = Differential(t)

mutable struct harmonic_oscillator <: HarmonicOscillatorBlox
    # all parameters are Num as to allow symbolic expressions
    ω::Num
    ζ::Num
    k::Num
    h::Num
    connector::Num
    odesystem::ODESystem
    function harmonic_oscillator(;name, ω=0.0, ζ=0.0, k=0.0, h=0.0)
        params = @parameters ω=ω ζ=ζ k=k h=h
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(ω, ζ, k, h, odesys.x, odesys)
    end
end

# This is for later to connect the icons to the different blox
# function gui.icon(Type::HarmonicOscillatorBlox)
#    return HarmonicOscillatorImage

mutable struct jansen_ritC <: JansenRitCBlox
    τ::Num
    H::Num
    λ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_ritC(;name, τ=0.0, H=0.0, λ=0.0, r=0.0)
        params = @parameters τ=τ H=H λ=λ r=r
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, H, λ, r, odesys.x, odesys)
    end
end

mutable struct jansen_ritSC <: JansenRitSCBlox
    τ::Num
    H::Num
    λ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_ritSC(;name, τ=0.0, H=0.0, λ=0.0, r=0.0)
        params = @parameters τ=τ H=H λ=λ r=r
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, H, λ, r, odesys.x, odesys)
    end
end

"""
Define canonical micro circuit according to the implementation in SPM12. Closesd representation is given in Bastos et al. 2015 
"A DCM study of spectral asymmetries in feedforward and feedback connections between visual areas V1 and V4 in the monkey"
Parameters values are taken from in this study.

TODO: 
- add the baseline subtraction to center firing rate at 0.
- connector is treated as the variable that represents the LFP, i.e. a measurement, not the variable that is used to connect blox. This needs to be re-designed.
"""
mutable struct cmc <: JansenRitBlox   # canonical micro circuit blox
    τ1::Num
    τ2::Num
    τ3::Num
    τ4::Num
    a11::Num
    a12::Num
    a13::Num
    a21::Num
    a22::Num
    a31::Num
    a33::Num
    a34::Num
    a43::Num
    a44::Num
    connector::Num
    odesystem::ODESystem
    function cmc(;name, τ1=0.002, τ2=0.002, τ3=0.016, τ4=0.028, a11=-800.0, a12=-800.0, a13=-800.0, a21=800.0, a22=-800.0, a31=800.0, a33=800.0, a34=400.0, a43=400.0, a44=-200.0)
        params = @parameters τ1=τ1 τ2=τ2 τ3=τ3 τ4=τ4
        sts    = @variables x1(t)=1.0 x2(t)=1.0 x3(t)=1.0 x4(t)=1.0 y1(t)=1.0 y2(t)=1.0 y3(t)=1.0 y4(t)=1.0
        σ(x)   = sigmoid(x, 2/3)   # slope parameter fixed, as in SPM12
        eqs    = [D(x1) ~ y1 - 2/τ1*x1,           # spiny stellate
                  D(y1) ~ -x1/τ1^2 + (a11*σ(x1) + a12*σ(x2) + a13*σ(x3))/τ1,
                  D(x2) ~ y2 - 2/τ2*x2,           # inhibitory interneuron
                  D(y2) ~ -x2/τ2^2 + (a21*σ(x1) + a22*σ(x2))/τ2,
                  D(x3) ~ y3 - 2/τ3*x3,           # deep pyramidal
                  D(y3) ~ -x3/τ3^2 + (a31*σ(x1) + a33*σ(x3) + a34*σ(x4))/τ3,
                  D(x4) ~ y4 - 2/τ4*x4,           # supragranular pyramidal
                  D(y4) ~ -x4/τ4^2 + (a43*σ(x3) + a44*σ(x4))/τ4]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ1, τ2, τ3, τ4, a11, a12, a13, a21, a22, a31, a33, a34, a43, a44, σ(odesys.x4) + σ(odesys.x3), odesys)
    end
end

mutable struct next_generation <: NextGenerationBlox
    C::Num
    Δ::Num
    η_0::Num
    v_syn::Num
    alpha_inv::Num
    k::Num
    connector::Num
    odesystem::ODESystem
    function next_generation(;name, C=0.0, Δ=0.0, η_0=0.0, v_syn=0.0, alpha_inv=0.0, k=0.0)
        params = @parameters C=C Δ=Δ η_0=η_0 v_syn=v_syn alpha_inv=alpha_inv k=k
        sts    = @variables Z(t)=0.5 g(t)=1.6
        Z = ModelingToolkit.unwrap(Z)
        g = ModelingToolkit.unwrap(g)
        C, Δ, η_0, v_syn, alpha_inv, k = map(ModelingToolkit.unwrap, [C, Δ, η_0, v_syn, alpha_inv, k])
        eqs = [Equation(D(Z), (1/C)*(-im*((Z-1)^2)/2 + (((Z+1)^2)/2)*(-Δ + im*(η_0) + im*v_syn*g) - ((Z^2-1)/2)*Z))
                    D(g) ~ alpha_inv*((k/(C*pi))*(1-abs(Z)^2)/(1+Z+conj(Z)+abs(Z)^2) - g)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(C, Δ, η_0, v_syn, alpha_inv, k, odesys.Z, odesys)
    end
end

