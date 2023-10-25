@parameters t
D = Differential(t)

mutable struct LinearNeuralMassBlox <: AbstractComponent
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
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function HarmonicOscillatorBlox(;name, ω=25*(2*pi), ζ=1.0, k=625*(2*pi), h=35.0)
        params = progress_scope(@parameters ω=ω ζ=ζ k=k h=h)
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        ω, ζ, k, h = params
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(odesys.x,[odesys.x],[odesys.x,odesys.y],
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
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitCBlox(;name, τ=0.001, H=20.0, λ=5.0, r=0.15)
        params = progress_scope(@parameters τ=τ H=H λ=λ r=r)
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        τ, H, λ, r = params
        eqs    = [D(x) ~ y - ((2/τ)*x),
                D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(odesys.x,[odesys.x],[odesys.x,odesys.y],
            Dict(odesys.x => (-1.0,1.0), odesys.y => (-1.0,1.0)),
            odesys)
    end
end
# this assignment is temporary until all the code is changed to the new name
const jansen_ritC = JansenRitCBlox

mutable struct  JansenRitSCBlox <: NeuralMassBlox
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
    function JansenRitSCBlox(;name, τ=0.014, H=20.0, λ=400.0, r=0.1)
        params = progress_scope(@parameters τ=τ H=H λ=λ r=r)
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        τ, H, λ, r = params
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(odesys.x,[odesys.x],[odesys.x,odesys.y],
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
    output
    connector::Num
    odesystem::ODESystem
    namespace
    function NextGenerationBlox(;name,namespace=nothing, C=30.0, Δ=1.0, η_0=5.0, v_syn=-10.0, alpha_inv=35.0, k=0.105)
        params = @parameters C=C Δ=Δ η_0=η_0 v_syn=v_syn alpha_inv=alpha_inv k=k
        sts    = @variables Z(t)=0.5 [output=true] g(t)=1.6
        Z = ModelingToolkit.unwrap(Z)
        g = ModelingToolkit.unwrap(g)
        C, Δ, η_0, v_syn, alpha_inv, k = map(ModelingToolkit.unwrap, [C, Δ, η_0, v_syn, alpha_inv, k])
        eqs = [Equation(D(Z), (1/C)*(-im*((Z-1)^2)/2 + (((Z+1)^2)/2)*(-Δ + im*(η_0) + im*v_syn*g) - ((Z^2-1)/2)*g))
                    D(g) ~ alpha_inv*((k/(C*pi))*(1-abs(Z)^2)/(1+Z+conj(Z)+abs(Z)^2) - g)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(C, Δ, η_0, v_syn, alpha_inv, k, sts[1], odesys.Z, odesys, namespace)
    end
end

mutable struct NextGenerationResolvedBlox <: NeuralMassBlox
    C::Num
    Δ::Num
    η_0::Num
    v_syn::Num
    alpha_inv::Num
    k::Num
    output
    connector::Num
    odesystem::ODESystem
    namespace
    function NextGenerationResolvedBlox(;name,namespace=nothing, C=30.0, Δ=1.0, η_0=5.0, v_syn=-10.0, alpha_inv=35.0, k=0.105)
        params = @parameters C=C Δ=Δ η_0=η_0 v_syn=v_syn alpha_inv=alpha_inv k=k
        sts    = @variables a(t)=0.5 [output=true] b(t)=0.0 [output=true] g(t)=1.6
        #Z = a + ib
        
        eqs = [ D(a) ~ (1/C)*(b*(a-1) - (Δ/2)*((a+1)^2-b^2) - η_0*b*(a+1) - v_syn*g*b*(a+1) - (g/2)*(a^2-b^2-1)),
                D(b) ~ (1/C)*((b^2-(a-1)^2)/2 - Δ*b*(a+1) + (η_0/2)*((a+1)^2-b^2) + v_syn*(g/2)*((a+1)^2-b^2) - a*b*g),
                D(g) ~ alpha_inv*((k/(C*pi))*((1-a^2-b^2)/(1+2*a+a^2+b^2)) - g)
               ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(C, Δ, η_0, v_syn, alpha_inv, k, sts[1], odesys.a, odesys, namespace)
    end
end


mutable struct NextGenerationEIBlox <: NeuralMassBlox
    Cₑ::Num
    Cᵢ::Num
    Δₑ::Num
    Δᵢ::Num
    η_0ₑ::Num
    η_0ᵢ::Num
    v_synₑₑ::Num
    v_synₑᵢ::Num
    v_synᵢₑ::Num
    v_synᵢᵢ::Num
    alpha_invₑₑ::Num
    alpha_invₑᵢ::Num
    alpha_invᵢₑ::Num
    alpha_invᵢᵢ::Num
    kₑₑ::Num
    kₑᵢ::Num
    kᵢₑ::Num
    kᵢᵢ::Num
    output
    connector::Num
    odesystem::ODESystem
    namespace
    function NextGenerationEIBlox(;name,namespace=nothing, Cₑ=30.0,Cᵢ=30.0, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0, alpha_invₑᵢ=0.8, alpha_invᵢₑ=10.0, alpha_invᵢᵢ=0.8, kₑₑ=0, kₑᵢ=0.5, kᵢₑ=0.65, kᵢᵢ=0)
        params = @parameters Cₑ=Cₑ Cᵢ=Cᵢ Δₑ=Δₑ Δᵢ=Δᵢ η_0ₑ=η_0ₑ η_0ᵢ=η_0ᵢ v_synₑₑ=v_synₑₑ v_synₑᵢ=v_synₑᵢ v_synᵢₑ=v_synᵢₑ v_synᵢᵢ=v_synᵢᵢ alpha_invₑₑ=alpha_invₑₑ alpha_invₑᵢ=alpha_invₑᵢ alpha_invᵢₑ=alpha_invᵢₑ alpha_invᵢᵢ=alpha_invᵢᵢ kₑₑ=kₑₑ kₑᵢ=kₑᵢ kᵢₑ=kᵢₑ kᵢᵢ=kᵢᵢ
        sts    = @variables aₑ(t)=-0.6 [output=true] bₑ(t)=0.18 [output=true] aᵢ(t)=0.02 [output=true] bᵢ(t)=0.21 [output=true] gₑₑ(t)=0 gₑᵢ(t)=0.23 gᵢₑ(t)=0.26 gᵢᵢ(t)=0
        
        #Z = a + ib
        
        eqs = [ D(aₑ) ~ (1/Cₑ)*(bₑ*(aₑ-1) - (Δₑ/2)*((aₑ+1)^2-bₑ^2) - η_0ₑ*bₑ*(aₑ+1) - (v_synₑₑ*gₑₑ+v_synₑᵢ*gₑᵢ)*(bₑ*(aₑ+1)) - (gₑₑ/2+gₑᵢ/2)*(aₑ^2-bₑ^2-1)),
                D(bₑ) ~ (1/Cₑ)*((bₑ^2-(aₑ-1)^2)/2 - Δₑ*bₑ*(aₑ+1) + (η_0ₑ/2)*((aₑ+1)^2-bₑ^2) + (v_synₑₑ*(gₑₑ/2)+v_synₑᵢ*(gₑᵢ/2))*((aₑ+1)^2-bₑ^2) - aₑ*bₑ*(gₑₑ+gₑᵢ)),
                D(aᵢ) ~ (1/Cᵢ)*(bᵢ*(aᵢ-1) - (Δᵢ/2)*((aᵢ+1)^2-bᵢ^2) - η_0ᵢ*bᵢ*(aᵢ+1) - (v_synᵢₑ*gᵢₑ+v_synᵢᵢ*gᵢᵢ)*(bᵢ*(aᵢ+1)) - (gᵢₑ/2+gᵢᵢ/2)*(aᵢ^2-bᵢ^2-1)),
                D(bᵢ) ~ (1/Cᵢ)*((bᵢ^2-(aᵢ-1)^2)/2 - Δᵢ*bᵢ*(aᵢ+1) + (η_0ᵢ/2)*((aᵢ+1)^2-bᵢ^2) + (v_synᵢₑ*(gᵢₑ/2)+v_synᵢᵢ*(gᵢᵢ/2))*((aᵢ+1)^2-bᵢ^2) - aᵢ*bᵢ*(gᵢₑ+gᵢᵢ)),
                D(gₑₑ) ~ alpha_invₑₑ*((kₑₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gₑₑ),
                D(gₑᵢ) ~ alpha_invₑᵢ*((kₑᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gₑᵢ),
                D(gᵢₑ) ~ alpha_invᵢₑ*((kᵢₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gᵢₑ),
                D(gᵢᵢ) ~ alpha_invᵢᵢ*((kᵢᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gᵢᵢ)
               ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(Cₑ, Cᵢ, Δₑ, Δᵢ, η_0ₑ, η_0ᵢ, v_synₑₑ, v_synₑᵢ, v_synᵢₑ, v_synᵢᵢ, alpha_invₑₑ, alpha_invₑᵢ, alpha_invᵢₑ, alpha_invᵢᵢ, kₑₑ, kₑᵢ, kᵢₑ, kᵢᵢ, sts[1], odesys.aₑ, odesys, namespace)
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

"""
New versions of blox begin here!
"""


"""
Units note: no units because no parameters :)
"""
struct LinearNeuralMass <: NeuralMassBlox
    output
    jcn
    odesystem
    namespace
    function LinearNeuralMass(;name, namespace=nothing)
        sts = @variables x(t) [output=true] jcn(t) [input=true]
        eqs = [D(x) ~ jcn]
        sys = System(eqs, name=name)
        new(sts[1], sts[2], sys, namespace)
    end
end

"""
Units note: Frequency should be tuned by user.
Updated with additional factors to make ms.
"""
struct HarmonicOscillator <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function HarmonicOscillator(;name, namespace=nothing, ω=25*(2*pi)*0.001, ζ=1.0, k=625*(2*pi), h=35.0)
        p = progress_scope(@parameters ω=ω ζ=ζ k=k h=h)
        sts    = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true]
        ω, ζ, k, h = p
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        sys = System(eqs, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
Units note: all units from the original Parkinson's paper EXCEPT τ. 
The original delays were in seconds, so multiplied to be consistent with other blocks in ms.
"""
# Constructing a new Jansen Rit blox to handle both delays and non-delays, along with default parameter inputs
struct JansenRit <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function JansenRit(;name,
                        namespace=nothing,
                        τ=nothing, 
                        H=nothing, 
                        λ=nothing, 
                        r=nothing, 
                        cortical=true)

        τ = isnothing(τ) ? (cortical ? 1 : 14) : τ
        H = isnothing(H) ? 20.0 : H # H doesn't have different parameters for cortical and subcortical
        λ = isnothing(λ) ? (cortical ? 5.0 : 400.0) : λ
        r = isnothing(r) ? (cortical ? 0.15 : 0.1) : r

        p = progress_scope(@parameters τ=τ H=H λ=λ r=r)
        τ, H, λ, r = p
        sts = @variables x(..)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true] 
        eqs = [D(x(t)) ~ y - ((2/τ)*x(t)),
               D(y) ~ -x(t)/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        sys = System(eqs, name=name)
        #can't use outputs because x(t) is Num by then
        #wrote inputs similarly to keep consistent
        new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
Units note: Unclear where the defaults come from (close but not quite Wilson-Cowan referenced in TVB and elsewhere).
They're on the same order of magnitude as the original parameters which are in ms, so good to go for now.
"""
struct WilsonCowan <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function WilsonCowan(;name,
                        namespace=nothing,
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
                        η=1.0
    )
        p = progress_scope(@parameters τ_E=τ_E τ_I=τ_I a_E=a_E a_I=a_I c_EE=c_EE c_IE=c_IE c_EI=c_EI c_II=c_II θ_E=θ_E θ_I=θ_I η=η)

        τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η = p
        sts = @variables E(t)=1.0 [output=true] I(t)=1.0 jcn(t)=0.0 [input=true] #P(t)=0.0
        eqs = [D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + η*(jcn)))), #old form: D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + P + η*(jcn)))),
               D(I) ~ -I/τ_I + 1/(1 + exp(-a_I*(c_EI*E - c_II*I - θ_I)))]
        sys = System(eqs, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
Units note: From Yamashita et al. paper, designed to be in ms. Good to go for now.
"""
struct LarterBreakspear <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function LarterBreakspear(;
                        name,
                        namespace=nothing,
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
                        C=0.35
    )
        p = progress_scope(@parameters C=C δ_VZ=δ_VZ T_Ca=T_Ca δ_Ca=δ_Ca g_Ca=g_Ca V_Ca=V_Ca T_K=T_K δ_K=δ_K g_K=g_K V_K=V_K T_Na=T_Na δ_Na=δ_Na g_Na=g_Na V_Na=V_Na V_L=V_L g_L=g_L V_T=V_T Z_T=Z_T Q_Vmax=Q_Vmax Q_Zmax=Q_Zmax IS=IS a_ee=a_ee a_ei=a_ei a_ie=a_ie a_ne=a_ne a_ni=a_ni b=b τ_K=τ_K ϕ=ϕ r_NMDA=r_NMDA)
        C, δ_VZ, T_Ca, δ_Ca, g_Ca, V_Ca, T_K, δ_K, g_K, V_K, T_Na, δ_Na, g_Na,V_Na, V_L, g_L, V_T, Z_T, Q_Vmax, Q_Zmax, IS, a_ee, a_ei, a_ie, a_ne, a_ni, b, τ_K, ϕ, r_NMDA = p
        
        sts = @variables V(t)=0.5 Z(t)=0.5 W(t)=0.5 jcn(t)=0.0 [input=true] Q_V(t) [output=true] Q_Z(t) m_Ca(t) m_Na(t) m_K(t)

        eqs = [ D(V) ~ -(g_Ca + (1 - C) * r_NMDA * a_ee * Q_V + C * r_NMDA * a_ee * jcn) * m_Ca * (V-V_Ca) -
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
        sys = System(eqs; name=name)
        new(p, sts[5], sts[4], sys, namespace)
    end
end
