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
        @show typeof(odesys)
        new(ω, ζ, k, h, odesys.x, odesys)
    end
end

# This is for later to connect the icons to the different blox
#function gui.icon(Type::HarmonicOscillatorBlox)
#    return HarmonicOscillatorImage

mutable struct jansen_rit <: JansenRitBlox
    τ::Num
    H::Num
    λ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit(;name, τ=0.0, H=0.0, λ=0.0, r=0.0)
        params = @parameters τ=τ H=H λ=λ r=r
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, H, λ, r, odesys.x, odesys)
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

