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
        new(Cₑ, Cᵢ, sts[1], odesys.aₑ, odesys, namespace)
    end
end
# this assignment is temporary until all the code is changed to the new name
const next_generation = NextGenerationBlox

"""
    LinearNeuralMass(name, namespace)

Create standard linear neural mass blox with a single internal state.
There are no parameters in this blox.
This is a blox of the sort used for spectral DCM modeling.
The formal definition of this blox is:


```math
\\frac{d}{dx} = \\sum{jcn}
```

where ``jcn``` is any input to the blox.


Arguments:
- name: Options containing specification about deterministic.
- namespace: Additional namespace above name if needed for inheritance.
"""

struct LinearNeuralMass <: NeuralMassBlox
    output
    jcn
    odesystem
    namespace
    function LinearNeuralMass(;name, namespace=nothing)
        sts = @variables x(t)=0.0 [output=true] jcn(t)=0.0 [input=true]
        eqs = [D(x) ~ jcn]
        sys = System(eqs, t, name=name)
        new(sts[1], sts[2], sys, namespace)
    end
end

"""
    HarmonicOscillator(name, namespace, ω, ζ, k, h)

    Create a harmonic oscillator blox with the specified parameters.
    The formal definition of this blox is:

```math
\\frac{dx}{dt} = y-(2*\\omega*\\zeta*x)+ k*(2/\\pi)*(atan((\\sum{jcn})/h)
\\frac{dy}{dt} = -(\\omega^2)*x
```
    where ``jcn`` is any input to the blox.
    

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- ω: Base frequency. Note the default value is scaled to give oscillations in milliseconds to match other blocks.
- ζ: Damping ratio.
- k: Gain.
- h: Threshold.
"""
struct HarmonicOscillator <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function HarmonicOscillator(;name, namespace=nothing, ω=25*(2*pi)*0.001, ζ=1.0, k=625*(2*pi), h=35.0)
        # p = progress_scope(ω, ζ, k, h)
        p = paramscoping(ω=ω, ζ=ζ, k=k, h=h)
        sts    = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true]
        ω, ζ, k, h = p
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        sys = System(eqs, t, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end


"""
    JansenRit(name, namespace, τ, H, λ, r, cortical, delayed)

    Create a Jansen Rit blox as described in Liu et al.
    The formal definition of this blox is:

```math
\\frac{dx}{dt} = y-\\frac{2}{\\tau}x
\\frac{dy}{dt} = -\\frac{x}{\\tau^2} + \\frac{H}{\\tau} [\\frac{2\\lambda}{1+\\text{exp}(-r*\\sum{jcn})} - \\lambda]
```

where ``jcn`` is any input to the blox.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- τ: Time constant. This is changed from the original source as the time constant was in seconds, while all our blocks are in milliseconds.
- H: See equation for use.
- λ: See equation for use.
- r: See equation for use.
- cortical: Boolean to determine whether to use cortical or subcortical parameters. Specifying any of the parameters above will override this.
- delayed: Boolean to indicate whether states are delayed

Citations:
1. Liu C, Zhou C, Wang J, Fietkiewicz C, Loparo KA. The role of coupling connections in a model of the cortico-basal ganglia-thalamocortical neural loop for the generation of beta oscillations. Neural Netw. 2020 Mar;123:381-392. doi: 10.1016/j.neunet.2019.12.021.

"""
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
                        cortical=true, 
                        delayed=false)

        τ = isnothing(τ) ? (cortical ? 1 : 14) : τ
        H = isnothing(H) ? 0.02 : H # H doesn't have different parameters for cortical and subcortical
        λ = isnothing(λ) ? (cortical ? 5.0 : 400.0) : λ
        r = isnothing(r) ? (cortical ? 0.15 : 0.1) : r

        # p = progress_scope(τ, H, λ, r)
        p = paramscoping(τ=τ, H=H, λ=λ, r=r)
        τ, H, λ, r = p
        if !delayed
            sts = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true] 
            eqs = [D(x) ~ y - ((2/τ)*x),
                   D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        else
            sts = @variables x(..)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true] 
            eqs = [D(x(t)) ~ y - ((2/τ)*x(t)),
                   D(y) ~ -x(t)/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        end
        sys = System(eqs, t, name=name)
        #can't use outputs because x(t) is Num by then
        #wrote inputs similarly to keep consistent
        return new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
    WilsonCown(name, namespace, τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η)

    Create a standard Wilson Cowan blox.
    The formal definition of this blox is:

```math
\\frac{dE}{dt} = \\frac{-E}{\\tau_E} + \\frac{1}{1 + \\text{exp}(-a_E*(c_{EE}*E - c_{IE}*I - \\theta_E + \\eta*(\\sum{jcn}))}
\\frac{dI}{dt} = \\frac{-I}{\\tau_I} + \\frac{1}{1 + exp(-a_I*(c_{EI}*E - c_{II}*I - \\theta_I)}
```
where ``jcn`` is any input to the blox.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Others: See equation for use.
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
        # p = progress_scope(τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η)
        p = paramscoping(τ_E=τ_E, τ_I=τ_I, a_E=a_E, a_I=a_I, c_EE=c_EE, c_IE=c_IE, c_EI=c_EI, c_II=c_II, θ_E=θ_E, θ_I=θ_I, η=η)
        τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η = p
        sts = @variables E(t)=1.0 [output=true] I(t)=1.0 jcn(t)=0.0 [input=true] #P(t)=0.0
        eqs = [D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + η*(jcn)))), #old form: D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + P + η*(jcn)))),
               D(I) ~ -I/τ_I + 1/(1 + exp(-a_I*(c_EI*E - c_II*I - θ_I)))]
        sys = System(eqs, t, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
    LarterBreakspear(name, namespace, ...)

    Create a Larter Breakspear blox described in Endo et al. For a full list of the parameters used see the reference.
    If you need to modify the parameters, see Chesebro et al. and van Nieuwenhuizen et al. for physiological ranges.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citations:
1. Endo H, Hiroe N, Yamashita O. Evaluation of Resting Spatio-Temporal Dynamics of a Neural Mass Model Using Resting fMRI Connectivity and EEG Microstates. Front Comput Neurosci. 2020 Jan 17;13:91. doi: 10.3389/fncom.2019.00091.
2. Chesebro AG, Mujica-Parodi LR, Weistuch C. Ion gradient-driven bifurcations of a multi-scale neuronal model. Chaos Solitons Fractals. 2023 Feb;167:113120. doi: 10.1016/j.chaos.2023.113120. 
3. van Nieuwenhuizen, H, Chesebro, AG, Polis, C, Clarke, K, Strey, HH, Weistuch, C, Mujica-Parodi, LR. Ketosis regulates K+ ion channels, strengthening brain-wide signaling disrupted by age. Preprint. bioRxiv 2023.05.10.540257; doi: https://doi.org/10.1101/2023.05.10.540257. 

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
        # p = progress_scope(C, δ_VZ, T_Ca, δ_Ca, g_Ca, V_Ca, T_K, δ_K, g_K, V_K, T_Na, δ_Na, g_Na, V_Na, V_L, g_L, V_T, Z_T, Q_Vmax, Q_Zmax, IS, a_ee, a_ei, a_ie, a_ne, a_ni, b, τ_K, ϕ,r_NMDA)
        p = paramscoping(C=C, δ_VZ=δ_VZ, T_Ca=T_Ca, δ_Ca=δ_Ca, g_Ca=g_Ca, V_Ca=V_Ca, T_K=T_K, δ_K=δ_K, g_K=g_K, V_K=V_K, T_Na=T_Na, δ_Na=δ_Na, g_Na=g_Na, V_Na=V_Na, V_L=V_L, g_L=g_L, V_T=V_T, Z_T=Z_T, Q_Vmax=Q_Vmax, Q_Zmax=Q_Zmax, IS=IS, a_ee=a_ee, a_ei=a_ei, a_ie=a_ie, a_ne=a_ne, a_ni=a_ni, b=b, τ_K=τ_K, ϕ=ϕ, r_NMDA=r_NMDA)
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
        sys = System(eqs, t; name=name)
        new(p, sts[5], sts[4], sys, namespace)
    end
end
