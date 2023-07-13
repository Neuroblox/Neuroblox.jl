@parameters t
D = Differential(t)

mutable struct LinearNeuralMassBlox <: NBComponent
    # τ::Num
    connector::Num
    odesystem::ODESystem
    function LinearNeuralMassBlox(;name)
        # params = @parameters τ=τ
        states = @variables x(t) jcn(t)
        eqs = D(x) ~ jcn
        odesys = ODESystem(eqs, t, states, []; name=name)
        new(odesys.x, odesys)
    end
end

mutable struct HarmonicOscillatorBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    ω::Num
    ζ::Num
    k::Num
    h::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function HarmonicOscillatorBlox(;name, ω=25*(2*pi), ζ=1.0, k=625*(2*pi), h=35.0)
        params = @parameters ω=ω ζ=ζ k=k h=h
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(ω, ζ, k, h, odesys.x,[odesys.x],[odesys.x,odesys.y],
            Dict(odesys.x => (-1.0,1.0), odesys.y => (-1.0,1.0)),
            odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const harmonic_oscillator = HarmonicOscillatorBlox

# This is for later to connect the icons to the different blox
# function gui.icon(Type::HarmonicOscillatorBlox)
#    return HarmonicOscillatorImage

mutable struct JansenRitCBlox <: NeuralMassBlox
    τ::Num
    H::Num
    λ::Num
    r::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitCBlox(;name, τ=0.001, H=20.0, λ=5.0, r=0.15)
        params = parameter_list(Dict(:τ => τ,:H => H,:λ => λ,:r => r))
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, H, λ, r, odesys.x,[odesys.x],[odesys.x,odesys.y],
            Dict(odesys.x => (-1.0,1.0), odesys.y => (-1.0,1.0)),
            odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const jansen_ritC = JansenRitCBlox

mutable struct  JansenRitSCBlox <: NeuralMassBlox
    τ::Num
    H::Num
    λ::Num
    r::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitSCBlox(;name, τ=0.014, H=20.0, λ=400.0, r=0.1)
        params = parameter_list(Dict(:τ => τ,:H => H,:λ => λ,:r => r))
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, H, λ, r, odesys.x,[odesys.x],[odesys.x,odesys.y],
            Dict(odesys.x => (-1.0,1.0), odesys.y => (-1.0,1.0)),
            odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const jansen_ritSC = JansenRitSCBlox

mutable struct WilsonCowanBlox <: NeuralMassBlox
    τ_E::Num
    τ_I::Num
    a_E::Num
    a_I::Num
    c_EE::Num
    c_EI::Num
    c_IE::Num
    c_II::Num
    θ_E::Num
    θ_I::Num
    η::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    odesystem::ODESystem
    function WilsonCowanBlox(;name,
                          τ_E=1.0,
                          τ_I=1.0,
                          a_E=1.2,
                          a_I=2.0,
                          c_EE=5.0,
                          c_IE=6.0,
                          c_EI=10.0,
                          c_II=1.0,
                          θ_E=2.0,
                          θ_I=3.5,
                          η=1.0)
        params = @parameters τ_E=τ_E τ_I=τ_I a_E=a_E a_I=a_I c_EE=c_EE c_IE=c_IE c_EI=c_EI c_II=c_II θ_E=θ_E θ_I=θ_I η=η
        sts    = @variables E(t)=1.0 I(t)=1.0 jcn(t)=0.0 P(t)=0.0
        eqs    = [D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + P + η*(jcn)))),
                  D(I) ~ -I/τ_I + 1/(1 + exp(-a_I*(c_EI*E - c_II*I - θ_I)))]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ_E,τ_I,a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η, odesys.E, [odesys.E],[odesys.E, odesys.I],odesys)
    end
end

mutable struct NextGenerationBlox <: NeuralMassBlox
    C::Num
    Δ::Num
    η_0::Num
    v_syn::Num
    alpha_inv::Num
    k::Num
    connector::Num
    odesystem::ODESystem
    function NextGenerationBlox(;name, C=30.0, Δ=1.0, η_0=5.0, v_syn=-10.0, alpha_inv=35.0, k=0.105)
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
# this assignment is temporary until all the code is changed to the new name
const next_generation = NextGenerationBlox

mutable struct LarterBreakspearBlox <: NeuralMassBlox
    C::Num
    δ_VZ::Num
    T_Ca::Num
    δ_Ca::Num
    g_Ca::Num
    V_Ca::Num
    T_K::Num
    δ_K::Num
    g_K::Num
    V_K::Num
    T_Na::Num
    δ_Na::Num
    g_Na::Num
    V_Na::Num
    V_L::Num
    g_L::Num
    V_T::Num
    Z_T::Num
    Q_Vmax::Num
    Q_Zmax::Num
    IS::Num
    a_ee::Num
    a_ei::Num
    a_ie::Num
    a_ne::Num
    a_ni::Num
    b::Num
    τ_K::Num
    ϕ::Num
    r_NMDA::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function LarterBreakspearBlox(;name,
                          T_Ca=-0.01,
                          δ_Ca=0.15,
                          g_Ca=1.0,
                          V_Ca=1.0,
                          T_K=0.0,
                          δ_K=0.3,
                          g_K=2.0,
                          V_K=-0.7,
                          T_Na=0.3,
                          δ_Na=0.15,
                          g_Na=6.7,
                          V_Na=0.53,
                          V_L=-0.5,
                          g_L=0.5,
                          V_T=0.0,
                          Z_T=0.0,
                          δ_VZ=0.61,
                          Q_Vmax=1.0,
                          Q_Zmax=1.0,
                          IS = 0.3,
                          a_ee=0.36,
                          a_ei=2.0,
                          a_ie=2.0,
                          a_ne=1.0,
                          a_ni=0.4,
                          b=0.1,
                          τ_K=1.0,
                          ϕ=0.7,
                          r_NMDA=0.25,
                          C=0.35)
        params = @parameters C=C δ_VZ=δ_VZ T_Ca=T_Ca δ_Ca=δ_Ca g_Ca=g_Ca V_Ca=V_Ca T_K=T_K δ_K=δ_K g_K=g_K V_K=V_K T_Na=T_Na δ_Na=δ_Na g_Na=g_Na V_Na=V_Na V_L=V_L g_L=g_L V_T=V_T Z_T=Z_T Q_Vmax=Q_Vmax Q_Zmax=Q_Zmax IS=IS a_ee=a_ee a_ei=a_ei a_ie=a_ie a_ne=a_ne a_ni=a_ni b=b τ_K=τ_K ϕ=ϕ r_NMDA=r_NMDA
        sts    = @variables V(t)=0.5 Z(t)=0.5 W(t)=0.5 jcn(t)=0.0 Q_V(t) Q_Z(t) m_Ca(t) m_Na(t) m_K(t)

        eqs    = [D(V) ~ -(g_Ca + (1 - C) * r_NMDA * a_ee * Q_V + C * r_NMDA * a_ee * jcn) * m_Ca * (V-V_Ca) -
                          g_K * W * (V - V_K) - g_L * (V - V_L) -
                          (g_Na * m_Na + (1 - C) * a_ee * Q_V + C * a_ee * jcn) * (V-V_Na) -
                          a_ie * Z * Q_Z + a_ne * IS,
                  D(Z) ~ b * (a_ni * IS + a_ei * V * Q_V),
                  D(W) ~ ϕ * (m_K - W) / τ_K,
                  Q_V ~ 0.5*Q_Vmax*(1 + tanh((V-V_T)/δ_VZ)),
                  Q_Z ~ 0.5*Q_Zmax*(1 + tanh((Z-Z_T)/δ_VZ)),
                  m_Ca ~  0.5*(1 + tanh((V-T_Ca)/δ_Ca)),
                  m_Na ~  0.5*(1 + tanh((V-T_Na)/δ_Na)),
                  m_K ~  0.5*(1 + tanh((V-T_K)/δ_K))]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(C, δ_VZ, T_Ca, δ_Ca, g_Ca, V_Ca, T_K, δ_K, g_K, V_K, T_Na, δ_Na, g_Na,V_Na, V_L,
        g_L, V_T, Z_T, Q_Vmax, Q_Zmax, IS, a_ee, a_ei, a_ie, a_ne, a_ni, b, τ_K, ϕ, r_NMDA,
        0.5*Q_Vmax*(1 + tanh((odesys.V-V_T)/δ_VZ)),
        [odesys.V],[odesys.V, odesys.Z, odesys.W],
        Dict(odesys.V => (-1.0,1.0), odesys.Z => (-1.0,1.0), odesys.W => (0.0,1.0)),
        odesys)
    end
end
