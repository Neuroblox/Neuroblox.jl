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
@blox struct LinearNeuralMass(; name, namespace=nothing) <: AbstractNeuralMass
    @params
    @states x = 0.0
    @inputs jcn = 0.0
    @outputs x
    @equations begin
        D(x) = jcn
    end
end

"""
    HarmonicOscillator(name, namespace, ω, ζ, k, h)

Create a harmonic oscillator blox with the specified parameters.

The formal definition of this blox is:

```math
\\frac{dx}{dt} = y-(2*\\omega*\\zeta*x)+ k*(2/\\pi)*(\\text{atan}((\\sum{jcn})/h) \\\\
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
@blox struct HarmonicOscillator(; name, namespace=nothing,
                                ω=25*(2*pi)*0.001, ζ=1.0, k=625*(2*pi), h=35.0) <: AbstractNeuralMass
    @params ω ζ k h
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
        D(y) = -(ω^2)*x
    end
end

"""
    JansenRit(name, namespace, τ, H, λ, r, cortical, delayed)

Create a Jansen Rit blox as described in Liu et al.
The formal definition of this blox is:

```math
\\frac{dx}{dt} = y-\\frac{2}{\\tau}x \\\\
\\frac{dy}{dt} = -\\frac{x}{\\tau^2} + \\frac{H}{\\tau} [\\frac{2\\lambda}{1 + e^{-r*\\sum{jcn}}} - \\lambda]
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
@blox struct JansenRit(; name,
                       namespace=nothing,
                       cortical=true,
                       τ=(cortical ? 1 : 14),
                       H=0.02,
                       λ=(cortical ? 5.0 : 400.0),
                       r=(cortical ? 0.15 : 0.1),
                       delayed=false) <: AbstractNeuralMass
    if delayed
        error("Delay systems are currently not supported")
    end
    @params τ H λ r
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y - ((2x/τ))
        D(y) = -x/(τ^2) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)
    end
end

"""
    WilsonCowan(name, namespace, τ_E, τ_I, a_E, a_I, c_EE, c_IE, c_EI, c_II, θ_E, θ_I, η)

Create a standard Wilson Cowan blox.
The formal definition of this blox is:

```math
\\frac{dE}{dt} = \\frac{-E}{\\tau_E} + \\frac{1}{1 + \\text{exp}(-a_E*(c_{EE}*E - c_{IE}*I - \\theta_E + \\eta*(\\sum{jcn}))} \\\\
\\frac{dI}{dt} = \\frac{-I}{\\tau_I} + \\frac{1}{1 + e^{-a_I*(c_{EI}*E - c_{II}*I) - \\theta_I}}
```
where ``jcn`` is any input to the blox.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- Others: See equation for use.
"""
@blox struct WilsonCowan(; name,
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
                        η=1.0) <: AbstractNeuralMass
    @params τ_E τ_I a_E a_I c_EE c_IE c_EI c_II θ_E θ_I η
    @states E=1.0 I=1.0
    @inputs jcn=0.0
    @outputs E
    @equations begin
        D(E) = -E/τ_E + 1/(1 + exp(-a_E*(c_EE*E - c_IE*I - θ_E + η*(jcn))))
        D(I) = -I/τ_I + 1/(1 + exp(-a_I*(c_EI*E - c_II*I - θ_I)))
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
@blox struct LarterBreakspear(;
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
                        C=0.35) <: AbstractNeuralMass
    @params C δ_VZ T_Ca δ_Ca g_Ca V_Ca T_K δ_K g_K V_K T_Na δ_Na g_Na V_Na V_L g_L V_T Z_T Q_Vmax Q_Zmax IS a_ee a_ei a_ie a_ne a_ni b τ_K ϕ r_NMDA
    @states V=0.5 Z=0.5 W=0.5
    @inputs jcn=0.0
    @outputs Q_Z
    @equations begin
        @setup begin
            (; m_Ca, m_Na, m_K, Q_V, Q_Z) = __sys__
        end
        D(V) = -(g_Ca + (1 - C) * r_NMDA * a_ee * Q_V + C * r_NMDA * a_ee * jcn) * m_Ca * (V-V_Ca) -
            g_K * W * (V - V_K) - g_L * (V - V_L) -
            (g_Na * m_Na + (1 - C) * a_ee * Q_V + C * a_ee * jcn) * (V-V_Na) -
            a_ie * Z * Q_Z + a_ne * IS
        D(Z) = b * (a_ni * IS + a_ei * V * Q_V)
        D(W) = ϕ * (m_K - W) / τ_K
    end
    @computed_properties begin
        Q_V  = 0.5*Q_Vmax*(1 + tanh((V-V_T)/δ_VZ))
        Q_Z  = 0.5*Q_Zmax*(1 + tanh((Z-Z_T)/δ_VZ))
        m_Ca = 0.5*(1 + tanh((V-T_Ca)/δ_Ca))
        m_Na = 0.5*(1 + tanh((V-T_Na)/δ_Na))
        m_K  = 0.5*(1 + tanh((V-T_K)/δ_K))
    end
    @computed_properties_with_inputs begin
        jcn = jcn
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
        \\dot{W} &= \\dfrac{d}{\\tau}\\,\\,(c V^2 + b V - \\beta W + a)
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
@blox struct Generic2dOscillator(;
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
                        bn=0.02) <: AbstractNeuralMass
    @params τ a b c d e f g α β γ bn
    @states V=0.0 W=1.0
    @inputs jcn=0.0
    @outputs V
    @equations begin
        D(V) = d * τ * ( -f * V^3 + e * V^2 + g * V + α * W - γ * jcn)
        D(W) = d / τ * ( c * V^2 + b * V - β * W + a)
    end
    @noise_equations begin
        W(V) = bn
        W(W) = bn
    end
end

"""
    KuramotoOscillator(name, namespace, ω, ζ, include_noise=false)

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
- include_noise: (default `false`) determines if brownian noise is included in the dynamics of the blox.
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
abstract type KuramotoOscillator <: AbstractNeuralMass end
function KuramotoOscillator(; name, namespace=nothing, ω=249.0, ζ=5.92, include_noise=false)
    if include_noise
        KuramotoOscillator_Noisy(;name, namespace, ω, ζ)
    else
        KuramotoOscillator_NonNoisy(;name, namespace, ω)
    end
end

@blox struct KuramotoOscillator_NonNoisy(; name, namespace=nothing, ω=249.0) <: KuramotoOscillator
    @params ω
    @states θ=0.0
    @inputs jcn=0.0
    @outputs θ
    @equations begin
        D(θ) = ω + jcn
    end
end

@blox struct KuramotoOscillator_Noisy(; name, namespace=nothing, ω=249.0, ζ=5.92) <: KuramotoOscillator
    @params ω ζ
    @states θ=0.0
    @inputs jcn=0.0
    @outputs θ
    @equations begin
        D(θ) = ω + jcn
    end
    @noise_equations begin
        W(θ) = ζ
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
- include_noise: (default `false`) determines if the system supports brownian noise
- ζ: (default `0.0`) strength of the Brownian noise term (if `include_noise == true`)
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citation:
1. Chen, L., & Campbell, S. A. (2022). Exact mean-field models for spiking neural networks with adaptation. Journal of Computational Neuroscience, 50(4), 445-469.
"""
abstract type NGNMM_Izh <: AbstractNeuralMass end
function NGNMM_Izh(; include_noise=false, kwargs...)
    if include_noise
        NGNMM_Izh_NonNoisy(; kwargs...)
    else
        NGNMM_Izh_Noisy(; kwargs...)
    end
end

@blox struct NGNMM_Izh_NonNoisy(; name, namespace=nothing, Δ=0.02, α=0.6215, gₛ=1.2308, η̄=0.12, I_ext=0.0, eᵣ=1.0, a=0.0077,
                                b=-0.0062, wⱼ=0.0189, sᵣ=1.2308, τₛ=2.6, κ=1.0) <: NGNMM_Izh
    @params Δ α gₛ η̄ I_ext eᵣ a b wⱼ sᵣ τₛ κ
    @states r=0.0 V=0.0 w=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        D(r) = Δ/π + 2*r*V - (α+gₛ*s*κ)*r
        D(V) = V^2 - α*V - w + η̄ + I_ext + gₛ*s*κ*(eᵣ - V) + jcn - (π*r)^2
        D(w) = a*(b*V - w) + wⱼ*r
        D(s) = -s/τₛ + sᵣ*r
    end
end

@blox struct NGNMM_Izh_Noisy(; name, namespace=nothing, Δ=0.02, α=0.6215, gₛ=1.2308, η̄=0.12, I_ext=0.0, eᵣ=1.0, a=0.0077,
                            b=-0.0062, wⱼ=0.0189, sᵣ=1.2308, τₛ=2.6, κ=1.0, ζ=0.0) <: NGNMM_Izh
    @params Δ α gₛ η̄ I_ext eᵣ a b wⱼ sᵣ τₛ κ ζ
    @states r=0.0 V=0.0 w=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        D(r) = Δ/π + 2*r*V - (α+gₛ*s*κ)*r
        D(V) = V^2 - α*V - w + η̄ + I_ext + gₛ*s*κ*(eᵣ - V) + jcn - (π*r)^2
        D(w) = a*(b*V - w) + wⱼ*r
        D(s) = -s/τₛ + sᵣ*r
    end
    @noise_equations begin
        W(V) = ζ
    end
end

"""
    NGNMM_QIF(name, namespace, ...)

This is the basic QIF next-gen neural mass as described in [1].
This includes the connections via firing rate as described in [1].

Equations:
    To be added once we have a final form that we like here.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- include_noise: (default `false`) determines if the system supports brownian noise
- A: (default `0.0`) strength of the Brownian noise term (if `include_noise == true`)
- Other parameters: See reference for full list. Note that parameters are scaled so that units of time are in milliseconds.

Citation:
Theta-nested gamma bursts by Torcini group.
"""
abstract type NGNMM_QIF <: AbstractNeuralMass end
function NGNMM_QIF(; kwargs...)
    if haskey(kwargs, :A)
        NGNMM_QIF_Noisy(;kwargs...)
    else
        NGNMM_QIF_NonNoisy(;kwargs...)
    end
end
@blox struct NGNMM_QIF_NonNoisy(; name, namespace=nothing, Δ=1.0, τₘ=20.0, H=1.3, I_ext=0.0, ω=0.0, J_internal=8.0) <: NGNMM_QIF
    @params Δ τₘ H I_ext ω J_internal
    @states r=0.0 V=0.0
    @inputs jcn=0.00
    @outputs r
    @equations begin
        D(r) = Δ/(π*τₘ^2) + 2*r*V/τₘ
        D(V) = (V^2 + H + I_ext*sin(ω*t))/τₘ - τₘ*(π*r)^2 + J_internal*r + jcn
    end
end
@blox struct NGNMM_QIF_Noisy(; name, namespace=nothing, Δ=1.0, τₘ=20.0, H=1.3, I_ext=0.0, ω=0.0, J_internal=8.0, A=0.0) <: NGNMM_QIF
    @params Δ τₘ H I_ext ω J_internal A
    @states r=0.0 V=0.0
    @inputs jcn=0.00
    @outputs r
    @equations begin
        D(r) = Δ/(π*τₘ^2) + 2*r*V/τₘ
        D(V) = (V^2 + H + I_ext*sin(ω*t))/τₘ - τₘ*(π*r)^2 + J_internal*r + jcn
    end
    @noise_equations begin
        W(V) = A
    end
end

"""
    VanDerPol(; name, namespace = nothing, θ=1.0, include_noise=false)

Create a neural mass model whose activity variable follows the dynamics of the (stochastic) van der Pol oscillator.

The formal definition of this blox is:
```math
\\frac{dx}{dt} = y
\\frac{dy}{dt} = θ(1 - x^2)y - x + ϕ ξ + jcn
```
where `jcn` is any input to the blox.

Arguments: 
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- `include_noise`: (default `false`) controls whether the system includes stochastic noise
- θ: damping strength
- ϕ: strength of the Brownian motion (if `include_noise == true`)
"""
abstract type VanDerPol <: AbstractNeuralMass end
function VanDerPol(; include_noise=false, kwargs...)
    if include_noise
        VanDerPol_Noisy(; kwargs...)
    else
        VanDerPol_NonNoisy(;kwargs...)
    end
end
@blox struct VanDerPol_NonNoisy(; name, namespace=nothing, θ=1.0) <: VanDerPol
    @params θ
    @states x=0.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = θ*(1-x^2)*y - x + jcn
    end
end

@blox struct VanDerPol_Noisy(; name, namespace=nothing, θ=1.0, ϕ=0.1) <: VanDerPol
    @params θ ϕ
    @states x=0.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = θ*(1-x^2)*y - x + jcn
    end
    @noise_equations begin
        W(y) = ϕ
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
@blox struct OUProcess(; name, namespace=nothing, μ=0.0, σ=1.0, τ=1.0) <: AbstractNeuralMass
    @params μ σ τ
    @states x=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = (-x + μ + jcn)/τ
    end
    @noise_equations begin
        W(x) = sqrt(2/τ)*σ
    end
end
