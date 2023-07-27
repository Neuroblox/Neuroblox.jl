@parameters t
D = Differential(t)

mutable struct LinearNeuralMassBlox <: NBComponent
    connector::Num
    odesystem::ODESystem
    function LinearNeuralMassBlox(;name)
        states = @variables x(t) jcn(t)
        eqs = D(x) ~ jcn
        odesys = ODESystem(eqs, t, states, []; name=name)
        new(odesys.x, odesys)
    end
end

mutable struct HarmonicOscillatorBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    p_dict::Dict{Symbol,Union{Real,Num}}
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function HarmonicOscillatorBlox(;name, ω=25*(2*pi), ζ=1.0, k=625*(2*pi), h=35.0)
        para_dict = scope_dict(Dict{Symbol,Union{Real,Num}}(:ω => ω,:ζ => ζ,:k => k,:h => h))
        ω=para_dict[:ω]
        ζ=para_dict[:ζ]
        k=para_dict[:k]
        h=para_dict[:h]
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        odesys = ODESystem(eqs, t, sts, values(para_dict); name=name)
        new(para_dict, odesys.x,[odesys.x],[odesys.x,odesys.y],
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
    p_dict::Dict{Symbol,Union{Real,Num}}
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitCBlox(;name, τ=0.001, H=20.0, λ=5.0, r=0.15)
        para_dict = scope_dict(Dict{Symbol,Union{Real,Num}}(:τ => τ,:H => H,:λ => λ,:r => r))
        τ=para_dict[:τ]
        H=para_dict[:H]
        λ=para_dict[:λ]
        r=para_dict[:r]
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, values(para_dict); name=name)
        new(para_dict, odesys.x,[odesys.x],[odesys.x,odesys.y],
            Dict(odesys.x => (-1.0,1.0), odesys.y => (-1.0,1.0)),
            odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const jansen_ritC = JansenRitCBlox

mutable struct  JansenRitSCBlox <: NeuralMassBlox
    p_dict::Dict{Symbol,Union{Real,Num}}
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitSCBlox(;name, τ=0.014, H=20.0, λ=400.0, r=0.1)
        para_dict = scope_dict(Dict{Symbol,Union{Real,Num}}(:τ => τ,:H => H,:λ => λ,:r => r))
        τ=para_dict[:τ]
        H=para_dict[:H]
        λ=para_dict[:λ]
        r=para_dict[:r]
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, values(para_dict); name=name)
        new(para_dict, odesys.x,[odesys.x],[odesys.x,odesys.y],
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
        eqs = [Equation(D(Z), (1/C)*(-im*((Z-1)^2)/2 + (((Z+1)^2)/2)*(-Δ + im*(η_0) + im*v_syn*g) - ((Z^2-1)/2)*g))
                    D(g) ~ alpha_inv*((k/(C*pi))*(1-abs(Z)^2)/(1+Z+conj(Z)+abs(Z)^2) - g)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(C, Δ, η_0, v_syn, alpha_inv, k, odesys.Z, odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const next_generation = NextGenerationBlox

mutable struct NextGenerationBloxCoupled <: NeuralMassBlox
    η_0e::Num
    η_0i::Num
    Δ_E::Num
    Δ_I::Num
    v_synE::Num
    v_synI::Num
    κ_EE::Num
    κ_EI::Num
    κ_IE::Num
    κ_II::Num
    α_invEE::Num
    α_invEI::Num
    α_invIE::Num
    α_invII::Num
    τ_E::Num
    τ_I::Num
    connector::Num
    odesystem::ODESystem
    function NextGenerationBloxCoupled(;name, η_0e=1.6, η_0i=1, Δ_E=0.2, Δ_I=0.2, v_synE=10, v_synI=-12, κ_EE=1, κ_EI=2, κ_IE=1.5, κ_II=2, α_invEE=3, α_invEI=3, α_invIE=10, α_invII=10, τ_E=12, τ_I=18)
        params = @parameters η_0e=η_0e η_0i=η_0i Δ_E=Δ_E Δ_I=Δ_I v_synE=v_synE v_synI=v_synI κ_EE=κ_EE κ_EI=κ_EI κ_IE=κ_IE κ_II=κ_II α_invEE=α_invEE α_invEI=α_invEI α_invIE=α_invIE α_invII=α_invII τ_E=τ_E τ_I=τ_I
        sts = @variables Z_E(t)=0.5 Z_I(t)=0.5 g_EE(t)=1.6 g_EI(t)=1.6 g_IE(t)=1.6 g_II(t)=1.6
        eqs = [D(Z_E) ~ (1/τ_E)*((-im*((Z_E-1)^2)/2)) #+ ((((Z_E^2+1)^2)/2)*(-Δ_E+(-im*η_0e)))) #+ ((im*((Z_E+1)^2)/2)*v_synE*g_EE) - (((Z_E^2-1)/2)*g_EE) + ((im*((Z_E+1)^2)/2)*v_synI*g_EI) - (((Z_E^2-1)/2)*g_EI))
               D(Z_I) ~ 0 #(1/τ_I)*((-im*((Z_I-1)^2)/2) + ((((Z_I^2+1)^2)/2)*(-Δ_I+(-im*η_0i)))) #+ ((im*((Z_I+1)^2)/2)*v_synE*g_IE) - (((Z_I^2-1)/2)*g_IE) + ((im*((Z_I+1)^2)/2)*v_synI*g_II) - (((Z_I^2-1)/2)*g_II))
               D(g_EE) ~ α_invEE*((κ_EE/(τ_E*pi))*(1-abs(Z_E)^2)/(1+Z_E+conj(Z_E)+abs(Z_E)^2) - g_EE)
               D(g_EI) ~ α_invEI*((κ_EI/(τ_I*pi))*(1-abs(Z_I)^2)/(1+Z_I+conj(Z_I)+abs(Z_I)^2) - g_EI)
               D(g_IE) ~ α_invIE*((κ_IE/(τ_E*pi))*(1-abs(Z_E)^2)/(1+Z_E+conj(Z_E)+abs(Z_E)^2) - g_IE)
               D(g_II) ~ α_invII*((κ_II/(τ_I*pi))*(1-abs(Z_I)^2)/(1+Z_I+conj(Z_I)+abs(Z_I)^2) - g_II)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(η_0e, η_0i, Δ_E, Δ_I, v_synE, v_synI, κ_EE, κ_EI, κ_IE, κ_II, α_invEE, α_invEI, α_invIE, α_invII, τ_E, τ_I, odesys.Z_E, odesys)
    end
end

# Notation from Chen and Campbell 2022
# Parameters come from Table 1 and Figure 1 caption
mutable struct NextGenerationBloxIz <: NeuralMassBlox
    α::Num
    Δ::Num
    η::Num
    I_ext::Num
    g_syn::Num
    a::Num
    s_jump::Num
    v_peak::Num
    τ_s::Num
    e_r::Num
    b::Num
    w_jump::Num
    v_reset::Num
    connector::Num
    odesystem::ODESystem
    function NextGenerationBloxIz(;name, α=0.6215, Δ=0.02, η=0.12, I_ext=0, g_syn=1.2308, a=0.0077, s_jump=1.2308, v_peak=200, τ_s=2.6, e_r=1, b=-0.0062, w_jump=0.0189, v_reset=-200)
        params = @parameters α=α g_syn=g_syn a=a s_jump=s_jump v_peak=v_peak τ_s=τ_s e_r=e_r b=b w_jump=w_jump v_reset=v_reset
        sts = @variables r(t)=0 v(t)=0 w(t)=0 s(t)=0
        eqs = [D(r) ~ Δ/π + 2*r*v - (α+g_syn*s)*r
                D(v) ~ v^2 - α*v - w + η + I_ext + g_syn*s*(e_r-v)-(π*r)^2
                D(w) ~ a*(b*v-w)+w_jump*r
                D(s) ~ -s/τ_s + s_jump*r]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(α, Δ, η, I_ext, g_syn, a, s_jump, v_peak, τ_s, e_r, b, w_jump, v_reset, odesys.v, odesys)
    end
end

# Old version to ask a question about during coding meeting
# mutable struct NextGenerationBloxEI <: NeuralMassBlox
#     η_0e::Num
#     η_0i::Num
#     Δ_E::Num
#     Δ_I::Num
#     v_synE::Num
#     v_synI::Num
#     κ_EE::Num
#     κ_EI::Num
#     κ_IE::Num
#     κ_II::Num
#     α_invEE::Num
#     α_invEI::Num
#     α_invIE::Num
#     α_invII::Num
#     τ_E::Num
#     τ_I::Num
#     connector::Num
#     odesystem::ODESystem
#     function NextGenerationBloxEI(;name, η_0e=1.6, η_0i=1, Δ_E=0.2, Δ_I=0.2, v_synE=10, v_synI=-12, κ_EE=1, κ_EI=2, κ_IE=1.5, κ_II=2, α_invEE=3, α_invEI=3, α_invIE=10, α_invII=10, τ_E=12, τ_I=18)
#         params = @parameters η_0e=η_0e η_0i=η_0i Δ_E=Δ_E Δ_I=Δ_I v_synE=v_synE v_synI=v_synI κ_EE=κ_EE κ_EI=κ_EI κ_IE=κ_IE κ_II=κ_II α_invEE=α_invEE α_invEI=α_invEI α_invIE=α_invIE α_invII=α_invII τ_E=τ_E τ_I=τ_I
#         sts = @variables Z_E(t)=0.5 Z_I(t)=0.5 g_EE(t)=1.6 g_EI(t)=1.6 g_IE(t)=1.6 g_II(t)=1.6
#         f(z, δ, η) = (-im*((z-1)^2)/2) + ((((z^2+1)^2)/2)*(-δ+(-im*η)))
#         g(z, g_xx, v_syn) = ((im*((z+1)^2)/2)*v_syn*g_xx) - (((z^2-1)/2)*g_xx)
#         eqs = [D(Z_E) ~ (1/τ_E)*(f(Z_E, Δ_E, η_0e) + g(Z_E, g_EE, v_synE) + g(Z_E, g_EI, v_synI))
#                D(Z_I) ~ (1/τ_I)*(f(Z_I, Δ_I, η_0i) + g(Z_I, g_IE, v_synE) + g(Z_I, g_II, v_synI))
#                D(g_EE) ~ α_invEE*((κ_EE/(τ_E*pi))*(1-abs(Z_E)^2)/(1+Z_E+conj(Z_E)+abs(Z_E)^2) - g_EE)
#                D(g_EI) ~ α_invEI*((κ_EI/(τ_I*pi))*(1-abs(Z_I)^2)/(1+Z_I+conj(Z_I)+abs(Z_I)^2) - g_EI)
#                D(g_IE) ~ α_invIE*((κ_IE/(τ_E*pi))*(1-abs(Z_E)^2)/(1+Z_E+conj(Z_E)+abs(Z_E)^2) - g_IE)
#                D(g_II) ~ α_invII*((κ_II/(τ_I*pi))*(1-abs(Z_I)^2)/(1+Z_I+conj(Z_I)+abs(Z_I)^2) - g_II)]
#         odesys = ODESystem(eqs, t, sts, params; name=name)
#         new(η_0e, η_0i, Δ_E, Δ_I, v_synE, v_synI, κ_EE, κ_EI, κ_IE, κ_II, α_invEE, α_invEI, α_invIE, α_invII, τ_E, τ_I, odesys.Z_E, odesys)
#     end
# end

# Primitive MPR (QIF) Next-Gen NMM blox for oscillation generation
mutable struct NextGenerationMPRBlox <: NeuralMassBlox
    Δ::Num
    η::Num
    J::Num
    I0::Num
    ω::Num
    connector::Num
    odesystem::ODESystem
    function NextGenerationMPRBlox(;name, Δ=1, η=-5, J=15, I0=3, ω=π/20)
        params = @parameters Δ=Δ η=η J=J I0 = I0 ω=ω
        sts = @variables r(t)=0 v(t)=-2
        eqs = [D(r) ~ (Δ/π) + 2*r*v
               D(v) ~ v^2 + η + J*r + I0*sin(ω*t)-(π*r)^2]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(Δ,η,J,I0,ω,odesys.v,odesys)
    end

end

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
