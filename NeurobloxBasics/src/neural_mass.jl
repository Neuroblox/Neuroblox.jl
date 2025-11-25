struct Noisy end
struct NonNoisy end

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
struct LinearNeuralMass <: AbstractNeuralMass
    system
    namespace

    function LinearNeuralMass(;name, namespace=nothing)
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        eqs = [D(x) ~ jcn]
        sys = System(eqs, t, name=name)
        new(sys, namespace)
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
struct HarmonicOscillator <: AbstractNeuralMass
    params
    system
    namespace

    function HarmonicOscillator(;name, namespace=nothing, ω=25*(2*pi)*0.001, ζ=1.0, k=625*(2*pi), h=35.0)
        # p = progress_scope(ω, ζ, k, h)
        p = paramscoping(ω=ω, ζ=ζ, k=k, h=h)
        sts    = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t) [input=true]
        ω, ζ, k, h = p
        eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
                  D(y) ~ -(ω^2)*x]
        sys = System(eqs, t, name=name)

        new(p, sys, namespace)
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
- τ: Time constant. Defaults to 1 for cortical regions, 14 for subcortical.
- H: See equation for use. Defaults to 0.02 for both cortical and subcortical regions.
- λ: See equation for use. Defaults to 5 for cortical regions, 400 for subcortical.
- r: See equation for use. Defaults to 0.15 for cortical regions, 0.1 for subcortical.
- cortical: Boolean to determine whether to use cortical or subcortical parameters. Specifying any of the parameters above will override this.
- delayed: Boolean to indicate whether states are delayed

Citations:
1. Liu C, Zhou C, Wang J, Fietkiewicz C, Loparo KA. The role of coupling connections in a model of the cortico-basal ganglia-thalamocortical neural loop for the generation of beta oscillations. Neural Netw. 2020 Mar;123:381-392. doi: 10.1016/j.neunet.2019.12.021.

"""
struct JansenRit <: AbstractNeuralMass
    params
    system
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
            sts = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t) [input=true] 
            eqs = [D(x) ~ y - ((2/τ)*x),
                   D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
            sys = System(eqs, t, name=name)
            #can't use outputs because x(t) is Num by then
            #wrote inputs similarly to keep consistent
            return new(p, sys, namespace)
        else
            error("Delay systems are currently not supported")
            # sts = @variables x(..)=1.0 [output=true] y(t)=1.0 jcn(t) [input=true] 
            # eqs = [D(x(t)) ~ y - ((2/τ)*x(t)),
            #        D(y) ~ -x(t)/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
            # sys = System(eqs, t, name=name)
            # #can't use outputs because x(t) is Num by then
            # #wrote inputs similarly to keep consistent
            # return new(p, sys, namespace)
        end
        sys = System(eqs, t, name=name)
        #can't use outputs because x(t) is Num by then
        #wrote inputs similarly to keep consistent
        return new(p, sts[1], sts[3], sys, namespace)
    end
end

"""
    WilsonCowan(name, namespace, τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η)

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
struct WilsonCowan <: AbstractNeuralMass
    params
    system
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
        sts = @variables E(t)=1.0 [output=true] I(t)=1.0 jcn(t) [input=true] #P(t)=0.0
        eqs = [D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + η*(jcn)))), #old form: D(E) ~ -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + P + η*(jcn)))),
               D(I) ~ -I/τ_I + 1/(1 + exp(-a_I*(c_EI*E - c_II*I - θ_I)))]
        sys = System(eqs, t, name=name)

        new(p, sys, namespace)
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
struct LarterBreakspear <: AbstractNeuralMass
    params
    system
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
        
        sts = @variables V(t)=0.5 Z(t)=0.5 W(t)=0.5 jcn(t) [input=true] Q_V(t) [output=true] Q_Z(t) m_Ca(t) m_Na(t) m_K(t)

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
        new(p, sys, namespace)
    end
end

"""
    Generic2dOscillator(name, namespace, ...)

The Generic2dOscillator model is a generic dynamic system with two state
variables. The dynamic equations of this model are composed of two ordinary
differential equations comprising two nullclines. The first nullcline is a
cubic function as it is found in most neuron and population models; the
second nullcline is arbitrarily configurable as a polynomial function up to
second order. The manipulation of the latter nullcline's parameters allows
to generate a wide range of different behaviours.

Equations:

```math
        \\begin{align}
        \\dot{V} &= d \\, \\tau (-f V^3 + e V^2 + g V + \\alpha W + \\gamma I) \\\\
        \\dot{W} &= \\dfrac{d}{\tau}\\,\\,(c V^2 + b V - \\beta W + a)
        \\end{align}
```

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citations:
FitzHugh, R., Impulses and physiological states in theoretical
models of nerve membrane, Biophysical Journal 1: 445, 1961.

Nagumo et.al, An Active Pulse Transmission Line Simulating
Nerve Axon, Proceedings of the IRE 50: 2061, 1962.

Stefanescu, R., Jirsa, V.K. Reduced representations of
heterogeneous mixed neural networks with synaptic coupling.
Physical Review E, 83, 2011.

Jirsa VK, Stefanescu R.  Neural population modes capture
biologically realistic large-scale network dynamics. Bulletin of
Mathematical Biology, 2010.

Stefanescu, R., Jirsa, V.K. A low dimensional description
of globally coupled heterogeneous neural networks of excitatory and
inhibitory neurons. PLoS Computational Biology, 4(11), 2008).

"""
struct Generic2dOscillator <: AbstractNeuralMass
    params
    system
    namespace

    function Generic2dOscillator(;
                        name,
                        namespace=nothing,
                        τ=1.0,
                        a=-2.0,
                        b=-10.0,
                        c=0.0,
                        d=0.02,
                        e=3.0,
                        f=1.0,
                        g=0.0,
                        α=1.0,
                        β=1.0,
                        γ=6e-2,
                        bn=0.02,
    )
        p = paramscoping(τ=τ, a=a,b=b,c=c,d=d,e=e,f=f,g=g,α=α,β=β,γ=γ)
        τ,a,b,c,d,e,f,g,α,β,γ = p
        
        sts = @variables V(t)=0.0 [output = true] W(t)=1.0 jcn(t) [input=true]
        @brownian w v
        eqs = [ D(V) ~ d * τ * ( -f * V^3 + e * V^2 + g * V + α * W - γ * jcn) + bn * w,
                D(W) ~ d / τ * ( c * V^2 + b * V - β * W + a) + bn * v]
        sys = System(eqs, t, sts, p; name=name)
        new(p, sys, namespace)
    end
end

"""
    KuramotoOscillator(name, namespace, ...)

Simple implementation of the Kuramoto oscillator as described in the original paper [1].
Useful for general models of synchronization and oscillatory behavior.
The general form of the Kuramoto oscillator is given by:
Equations:

```math
        \\begin{equation}
        \\dot{\\theta_i} = \\omega_i + \\frac{1}{N}\\sum_{j=1}^N{K_{i, j}\\text{sin}(\\theta_j - \\theta_i)}
        \\end{equation}
```

Where this describes the connection between regions \$i\$ and \$j\$. An alternative form
which includes a noise term for each region is also provided, taking the form:

```math
        \\begin{equation}
        \\dot{\\theta_i} = \\omega_i + \\zeta dW_i \\frac{1}{N}\\sum_{j=1}^N{K_{i, j}\\text{sin}(\\theta_j - \\theta_i)}
        \\end{equation}
```

where \$W_i\$ is a Wiener process and \$\\zeta_i\$ is the noise strength.

Keyword arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- `include_noise` (default `false`) determines if brownian noise is included in the dynamics of the blox.
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.
                    Default parameter values are taken from [2].

Citations:
1. Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. 
   In: Araki, H. (eds) International Symposium on Mathematical Problems in Theoretical Physics. 
   Lecture Notes in Physics, vol 39. Springer, Berlin, Heidelberg. https://doi.org/10.1007/BFb0013365

2. Sermon JJ, Wiest C, Tan H, Denison T, Duchet B. Evoked resonant neural activity long-term 
   dynamics can be reproduced by a computational model with vesicle depletion. Neurobiol Dis. 
   2024 Jun 14;199:106565. doi: 10.1016/j.nbd.2024.106565. Epub ahead of print. PMID: 38880431.

"""
struct KuramotoOscillator{IsNoisy} <: AbstractNeuralMass
    params
    system
    namespace

    function KuramotoOscillator(; name,
                                  namespace=nothing,
                                  ω=249.0,
                                  ζ=5.92,
                                  include_noise=false)
        if include_noise
            KuramotoOscillator{Noisy}(;name, namespace, ω, ζ)
        else
            KuramotoOscillator{NonNoisy}(;name, namespace, ω)
        end
    end
    function KuramotoOscillator{Noisy}(;name, namespace=nothing, ω=249.0, ζ=5.92)
        p = paramscoping(ω=ω, ζ=ζ)
        ω, ζ = p
        sts = @variables θ(t)=0.0 [output = true] jcn(t) [input=true]
        @brownian w
        eqs = [D(θ) ~ ω + jcn + ζ*w]
        sys = System(eqs, t, sts, p; name=name)
        new{Noisy}(p, sys, namespace)
    end
    function KuramotoOscillator{NonNoisy}(;name, namespace=nothing, ω=249.0)
        p = paramscoping(ω=ω)
        ω = p[1]
        sts = @variables θ(t)=0.0 [output = true] jcn(t) [input=true]
        eqs = [D(θ) ~ ω + jcn]
        sys = System(eqs, t, sts, p; name=name)
        new{NonNoisy}(p, sys, namespace)
    end
end

"""
    NGNMM_Izh(name, namespace, ...)

This is the basic Izhikevich next-gen neural mass as described in [1].
The corresponding connector is set up to allow for connections between masses, but the
user must add their own \$ \\kappa \$ values to the connection weight as there is no
good way of accounting for this weight within/between regions.

Currently, the connection weights include the presynaptic \$ g_s \$, but this could be changed.

Equations:
    To be added once we have a final form that we like here.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citation:
1. Chen, L., & Campbell, S. A. (2022). Exact mean-field models for spiking neural networks with adaptation. Journal of Computational Neuroscience, 50(4), 445-469.

"""
struct NGNMM_Izh{IsNoisy} <: AbstractNeuralMass
    params
    system
    namespace

    function NGNMM_Izh(;
                name,
                namespace=nothing,
                Δ=0.02,
                α=0.6215,
                gₛ=1.2308,
                η̄=0.12,
                I_ext=0.0,
                eᵣ=1.0,
                a=0.0077,
                b=-0.0062,
                wⱼ=0.0189,
                sⱼ=1.2308,
                τₛ=2.6,
                κ=1.0,
                ζ=0.0)
        if ζ == 0
            NGNMM_Izh{NonNoisy}(; name, namespace, Δ, α, gₛ, η̄, I_ext, eᵣ, a, b, wⱼ, sⱼ, τₛ, κ)
        else
            NGNMM_Izh{Noisy}(; name, namespace, Δ, α, gₛ, η̄, I_ext, eᵣ, a, b, wⱼ, sⱼ, τₛ, κ, ζ)
        end

    end
    function NGNMM_Izh{NonNoisy}(; name, namespace=nothing, Δ=0.02, α=0.6215, gₛ=1.2308, η̄=0.12, I_ext=0.0, eᵣ=1.0, a=0.0077, b=-0.0062, wⱼ=0.0189, sⱼ=1.2308, τₛ=2.6, κ=1.0)
        p = paramscoping(Δ=Δ, α=α, gₛ=gₛ, η̄=η̄, I_ext=I_ext, eᵣ=eᵣ, a=a, b=b, wⱼ=wⱼ, sⱼ=sⱼ, κ=κ)
        Δ, α, gₛ, η̄, I_ext, eᵣ, a, b, wⱼ, sⱼ, κ = p
        sts = @variables r(t)=0.0 V(t)=0.0 w(t)=0.0 s(t)=0.0 [output=true] jcn(t) [input=true]
        eqs = [ D(r) ~ Δ/π + 2*r*V - (α+gₛ*s*κ)*r,
                D(V) ~ V^2 - α*V - w + η̄ + I_ext + gₛ*s*κ*(eᵣ - V) + jcn - (π*r)^2,
                D(w) ~ a*(b*V - w) + wⱼ*r,
                D(s) ~ -s/τₛ + sⱼ*r
              ]
        sys = System(eqs, t, sts, p; name=name)
        new(p, sys, namespace)
    end
    function NGNMM_Izh{Noisy}(; name, namespace=nothing, Δ=0.02, α=0.6215, gₛ=1.2308, η̄=0.12, I_ext=0.0, eᵣ=1.0, a=0.0077, b=-0.0062, wⱼ=0.0189, sⱼ=1.2308, τₛ=2.6, κ=1.0, ζ=0.0)
        p = paramscoping(Δ=Δ, α=α, gₛ=gₛ, η̄=η̄, I_ext=I_ext, eᵣ=eᵣ, a=a, b=b, wⱼ=wⱼ, sⱼ=sⱼ, τₛ=τₛ, κ=κ, ζ=ζ)
        Δ, α, gₛ, η̄, I_ext, eᵣ, a, b, wⱼ, sⱼ, τₛ, κ, ζ = p
        sts = @variables r(t)=0.0 V(t)=0.0 w(t)=0.0 s(t)=0.0 [output=true] jcn(t) [input=true]
        @brownian ξ
        eqs = [ D(r) ~ Δ/π + 2*r*V - (α+gₛ*s*κ)*r,
                D(V) ~ V^2 - α*V - w + η̄ + I_ext + gₛ*s*κ*(eᵣ - V) + ζ*ξ + jcn - (π*r)^2,
                D(w) ~ a*(b*V - w) + wⱼ*r,
                D(s) ~ -s/τₛ + sⱼ*r
              ]
        sys = System(eqs, t, sts, p; name=name)
        new{Noisy}(p, sys, namespace)
    end
end

"""
    NGNMM_QIF(name, namespace, ...)

This is the basic QIF next-gen neural mass as described in [1].
This includes the connections via firing rate as described in [1] and the optional noise term.

Equations:
    To be added once we have a final form that we like here.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citation:
Theta-nested gamma bursts by Torcini group.
"""
struct NGNMM_QIF{IsNoisy} <: AbstractNeuralMass
    params
    system
    namespace

    function NGNMM_QIF(;
                        name,
                        namespace=nothing,
                        Δ=1.0,
                        τₘ=20.0,
                        H=1.3,
                        I_ext=0.0,
                        ω=0.0,
                        J_internal=8.0,
                        A=0.0)
        if A == 0
            NGNMM_QIF{NonNoisy}(; name, namespace, Δ, τₘ, H, I_ext, ω, J_internal)
        else
            NGNMM_QIF{Noisy}(; name, namespace, Δ, τₘ, H, I_ext, ω, J_internal, A)
        end
    end

    function NGNMM_QIF{NonNoisy}(; name, namespace=nothing, Δ=1.0, τₘ=20.0, H=1.3, I_ext=0.0, ω=0.0, J_internal=8.0)
        p = paramscoping(Δ=Δ, τₘ=τₘ, H=H, I_ext=I_ext, J_internal=J_internal)
        Δ, τₘ, H, I_ext, J_internal = p
        sts = @variables r(t)=0.0 [output=true] V(t)=0.0 jcn(t) [input=true]
        eqs = [D(r) ~ Δ/(π*τₘ^2) + 2*r*V/τₘ,
               D(V) ~ (V^2 + H + I_ext*sin(ω*t))/τₘ - τₘ*(π*r)^2 + J_internal*r  + jcn]
        sys = System(eqs, t, sts, p; name=name)

        new{NonNoisy}(p, sys, namespace)
    end

    function NGNMM_QIF{Noisy}(; name, namespace=nothing, Δ=1.0, τₘ=20.0, H=1.3, I_ext=0.0, ω=0.0, J_internal=8.0, A=0.0)
        p = paramscoping(Δ=Δ, τₘ=τₘ, H=H, I_ext=I_ext, J_internal=J_internal)
        Δ, τₘ, H, I_ext, J_internal = p
        sts = @variables r(t)=0.0 [output=true] V(t)=0.0 jcn(t) [input=true]
        @brownian ξ
        eqs = [D(r) ~ Δ/(π*τₘ^2) + 2*r*V/τₘ,
               D(V) ~ (V^2 + H + I_ext*sin(ω*t))/τₘ - τₘ*(π*r)^2 + J_internal*r  + A*ξ + jcn]
        sys = System(eqs, t, sts, p; name=name)

        new{Noisy}(p, sys, namespace)
    end
end

"""
    VanDerPol(; name, namespace = nothing, θ=1.0, ϕ=0.1, include_noise = false)

Create a neural mass model whose activity variable follows the dynamics of the (stochastic) van der Pol oscillator.

The formal definition of this blox is:
```math
\\frac{dx}{dt} = y
\\frac{dy}{dt} = θ(1 - x^2)y - x + ϕξ + jcn
```
where `jcn` is any input to the blox, and ξ is a Brownian variable that is added if `include_noise` is set.

Arguments: 
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- θ: damping strength
- ϕ: strength of the Brownian motion
"""
struct VanDerPol{IsNoisy} <: AbstractNeuralMass
    params
    system
    namespace

    function VanDerPol(; name, namespace=nothing, θ=1.0, ϕ=0.1, include_noise=false)
        if include_noise
            VanDerPol{Noisy}(;name, namespace, θ, ϕ)
        else
            VanDerPol{NonNoisy}(;name, namespace, θ)
        end
    end
    function VanDerPol{Noisy}(; name, namespace=nothing, θ=1.0, ϕ=0.1)
        p = paramscoping(θ=θ, ϕ=ϕ)
        θ, ϕ = p
        sts = @variables x(t)=0.0 [output=true] y(t)=0.0 jcn(t) [input=true]
        @brownian ξ

        eqs = [D(x) ~ y,
               D(y) ~ θ*(1-x^2)*y - x + ϕ*ξ + jcn]

        sys = System(eqs, t, sts, p; name=name)
        new{Noisy}(p, sys, namespace)
    end
    function VanDerPol{NonNoisy}(; name, namespace=nothing, θ=1.0)
        p = paramscoping(θ=θ)
        θ = p[1]
        sts = @variables x(t)=0.0 [output=true] y(t)=0.0 jcn(t) [input=true]
        
        eqs = [D(x) ~ y,
               D(y) ~ θ*(1-x^2)*y - x + jcn]

        sys = System(eqs, t, sts, p; name=name)
        new{NonNoisy}(p, sys, namespace)
    end
end

"""
    OUProcess(; name, namespace, μ, σ, τ)

Create a neural mass model whose activity variable follows an Ornstein-Uhlenbeck process.

The formal definition of this blox is:
```math
\\frac{dx}{dt} = (-x + μ + jcn)/τ + \\sqrt{2 / τ} σw
```
Where `w` is a Brownian variable and `jcn` is the input to the blox.

Arguments: 
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- τ: relaxation time
- μ: Mean of the OU process
- σ: Strength of the Brownian motion (variance of OUProcess process is τ*σ^2/2)
"""
mutable struct OUProcess <: AbstractNeuralMass
    # all parameters are Num as to allow symbolic expressions
    namespace
    stochastic
    system
    function OUProcess(;name, namespace=nothing, μ=0.0, σ=1.0, τ=1.0)
        p = paramscoping(μ=μ, τ=τ, σ=σ)
        μ, τ, σ = p
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        @brownian w

        eqs = [D(x) ~ (-x + μ + jcn)/τ + sqrt(2/τ)*σ*w]
        sys = System(eqs, t; name=name)
        new(namespace, true, sys)
    end
end
