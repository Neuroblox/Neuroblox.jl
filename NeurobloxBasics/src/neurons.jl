"""
    IFNeuron(name, namespace, C, θ, Eₘ, I_in)

Create a basic integrate-and-fire neuron.
This follows Lapicque's equation (see Abbott [1], with parameters chosen to match the LIF/QIF neurons implemented as well):

```math
\\frac{dV}{dt} = \\frac{I_{in} + jcn}{C}
```
where ``jcn`` is any input to the blox.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- C: Membrane capicitance (Default: 1 μF).
- θ: Threshold voltage (Default: -50 mV).
- Eₘ: Resting membrane potential (Default: -70 mV).
- I_in: External current input (Default: 0 μA).
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.05 ms).

References:
1. Abbott, L. Lapicque's introduction of the integrate-and-fire model neuron (1907). Brain Res Bull 50, 303-304 (1999).
"""
@blox struct IFNeuron(;name,
					  namespace=nothing, 
					  C = 1.0,
					  θ = -50.0,
					  Eₘ= -70.0,
					  I_in=0,
                      dtmax=0.05) <: AbstractNeuron
    # Parameter bounds for GUI
    # C = [0.1, 100] μF
    # θ = [-65, -45] mV
    # Eₘ = [-100, -55] mV - If Eₘ >= θ obvious instability
    # I_in = [-2.5, 2.5] μA
    # Remember: synaptic weights need to be in μA/mV, so they're very small!
    @params C θ Eₘ I_in dtmax
    @states V=-70.0
    @inputs jcn=0.0
    @outputs V
    @equations begin
        D(V) = (I_in + jcn)/C
    end
    @discrete_events (V >= θ) => (V=Eₘ,)
end

"""
    LIFNeuron(name, namespace, C, θ, Eₘ, I_in)

Create a leaky integrate-and-fire neuron.

This largely follows the formalism and parameters given in Chapter 8 of Sterratt et al. [1], with the following equations:

```math
\\frac{dV}{dt} = \\frac{\\frac{-(V-E_m)}{R_m} + I_{in} + jcn}{C}
\\frac{dG}{dt} = -\\frac{1}{\\tau}G
```

where ``jcn`` is any synaptic input to the blox (presumably a current G from another neuron).

Keyword Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- C: Membrane capicitance (Default: 1 μF).
- Eₘ: Resting membrane potential (Default -70 mV).
- Rₘ: Membrane resistance (Default: 10 kΩ).
- τ: Synaptic time constant (Default 10 ms).
- θ: Threshold voltage (Default: -50 mV).
- E_syn: Synaptic reversal potential (Default: -70 mV).
- G_syn: Synaptic conductance (Default: 0.002 μA/mV).
- I_in: External current input (Default: 0 μA).
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.05 ms).

References:
1. Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). Principles of Computational Modelling in Neuroscience. Cambridge University Press.
"""
@blox struct LIFNeuron(;name,
				       namespace=nothing, 
				       C=1.0,
				       Eₘ = -70.0,
				       Rₘ = 10.0,
				       τ = 10.0,
				       θ = -50.0,
				       E_syn=-70.0,
				       G_syn=0.002,
				       I_in=0.0,
                       dtmax=0.05) <: AbstractNeuron
    @params C Eₘ Rₘ τ θ E_syn G_syn I_in dtmax
    @states V=-70.0 G=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = (-(V-Eₘ)/Rₘ + I_in + jcn)/C
		D(G) = (-1/τ)*G
    end
    @discrete_events (V >= θ) => (V=Eₘ,
                                  G=G+G_syn)
end

"""
    LIFInhNeuron(; name, namespace, kwargs...)

Create an inhibitory leaky integrate-and-fire neuron. This is a model that uses LIF equations for the voltage dynamics, but explicitly models the gating dynamics of AMPA and GABA receptors and their synaptic decay.

The formal definition of this blox is:
```math
\\frac{dV}{dt} = (1 - \\mathbb{1}_\\text{refrac}) (-g_L(V - V_L) - S_\\text{AMPA, ext} g_\\text{AMPA, ext} (V - V_E) - S_\\text{GABA} g_\\text{GABA}(V - V_I) - S_\\text{AMPA} g_\\text{AMPA} (V - V_E) - jcn) / C \\\\
D(S_\\text{AMPA}) = - S_\\text{AMPA} / τ_\\text{AMPA} \\\\
D(S_\\text{GABA}) = - S_\\text{GABA} / τ_\\text{GABA} \\\\
D(S_\\text{AMPA, ext}) = - S_\\text{AMPA, ext} / τ_\\text{AMPA}
```

Arguments:
- g_L: Leak conductance (mS).
- V_L: Leak reversal potential (mV).
- V_E: Excitatory reversal potential (mV).
- V_I: Inhibitory reversal potential (mV).
- θ: Firing threshold potential (mV).
- V_reset: Reset potential after a spike (mV).
- C: Membrane capacitance (μF).
- τ_AMPA: time constant for the closing of AMPA receptors (ms).
- τ_GABA: time constant for the closing of GABA receptors (ms).
- t_refract: Refractory period after a spike (ms).
- α: Scaling for the rise of NMDA current (ms^-1).
- g_AMPA: Synaptic conductance for AMPA glutamate receptors (mS).
- g_AMPA_ext: Synaptic conductance for external current input through AMPA receptors (mS).
- g_GABA: Synaptic conductance for GABA glutamate receptors (mS).
- g_NMDA: Synaptic conductance for NMDA glutamate receptors (mS).
- Mg: Magnesium ion concentration (mM).
- exci_scaling_factor: Excitatory scaling factor.
- inh_scaling_factor: Inhibitory scaling factor.
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.05 ms).
"""
@blox struct LIFInhNeuron(
    ;name,
    namespace = nothing,
    g_L = 20 * 1e-6, # mS
    V_L = -70, # mV
    V_E = 0, # mV
    V_I = -70, # mV
    θ = -50, # mV
    V_reset = -55, # mV
    C = 0.2 * 1e-3, # mS / kHz 
    τ_AMPA = 2, # ms
    τ_GABA = 5, # ms
    t_refract = 1, # ms
    α = 0.5, # ms⁻¹
    g_AMPA = 0.04 * 1e-6, # mS
    g_AMPA_ext = 1.62 * 1e-6, # mS
    g_GABA = 1 * 1e-6, # mS
    g_NMDA = 0.13 * 1e-6, # mS 
    Mg = 1, # mM
    exci_scaling_factor = 1,
    inh_scaling_factor = 1,
    dtmax=0.05) <: AbstractInhNeuron
    
    @params( 
        g_L,
        V_L, 
        V_E,
        V_I,
        V_reset,
        θ,
        C,
        τ_AMPA, 
        τ_GABA, 
        t_refract_duration=t_refract,
        t_refract_end=-Inf,
        g_AMPA = g_AMPA * exci_scaling_factor,
        g_AMPA_ext = g_AMPA_ext,
        g_GABA = g_GABA * inh_scaling_factor,
        g_NMDA = g_NMDA * exci_scaling_factor,
        α=α,
        Mg=Mg,
        is_refractory=0,
        dtmax
    )
    @states(
        V=-52.0,
        S_AMPA=0.0,
        S_GABA=0.0,
        S_AMPA_ext=0.0
    )
    @inputs jcn=0.0 jcn_external=0.0
    @outputs V
    @equations begin
        D(V) = (1 - is_refractory) * (- g_L * (V - V_L) - S_AMPA_ext * g_AMPA_ext * (V - V_E) - S_GABA * g_GABA * (V - V_I) - S_AMPA * g_AMPA * (V - V_E) - jcn) / C
        D(S_AMPA) = - S_AMPA / τ_AMPA
        D(S_GABA) = - S_GABA / τ_GABA
        D(S_AMPA_ext) = - S_AMPA_ext / τ_AMPA
    end    
end

"""
    LIFExciNeuron(; name, namespace, kwargs...)

Create an excitatory leaky integrate-and-fire neuron. This is a model that uses LIF equations for the voltage dynamics, but explicitly models the gating dynamics of glutamate and GABA receptors and their synaptic decay.

The formal definition of this blox is:
```math
\\frac{dV}{dt} = (-g_L(V - V_L) - S_\\text{AMPA, ext} g_\\text{AMPA, ext} (V - V_E) - S_\\text{GABA} g_\\text{GABA}(V - V_I) - S_\\text{AMPA} g_\\text{AMPA} (V - V_E) - jcn) / C
D(S_\\text{AMPA}) = - S_\\text{AMPA} / τ_\\text{AMPA}\\\\
D(S_\\text{GABA}) = - S_\\text{GABA} / τ_\\text{GABA}\\\\
D(S_\\text{NMDA}) = - S_\\text{NMDA} / τ_\\text{NMDA, decay} + αx(1 - S_\\text{NMDA}) \\\\
\\frac{dx}{dt} = -x / τ_\\text{NMDA, rise} \\\\
\\frac{dS_\\text{AMPA, ext}}{dt} = -S_\\text{AMPA, ext} / τ_\\text{AMPA}
```

Keyword Arguments:
- g_L: Leak conductance (mS).
- V_L: Leak reversal potential (mV).
- V_E: Excitatory reversal potential (mV).
- V_I: Inhibitory reversal potential (mV).
- θ: Firing threshold potential (mV).
- V_reset: Reset potential after a spike (mV).
- C: Membrane capacitance (μF).
- τ_AMPA: time constant for the closing of AMPA receptors (ms).
- τ_GABA: time constant for the closing of GABA receptors (ms).
- τ_NMDA_decay: time constant for the closing of NMDA receptors (ms).
- τ_NMDA_rise: time constant for the decay of NMDA current post spike (ms).
- t_refract: Refractory period after a spike (ms).
- α: Scaling for the rise of NMDA current (ms^-1).
- g_AMPA: Synaptic conductance for AMPA glutamate receptors (mS).
- g_AMPA_ext: Synaptic conductance for external current input through AMPA receptors (mS).
- g_GABA: Synaptic conductance for GABA glutamate receptors (mS).
- g_NMDA: Synaptic conductance for NMDA glutamate receptors (mS).
- Mg: Magnesium ion concentration (mM).
- exci_scaling_factor: Excitatory scaling factor.
- inh_scaling_factor: Inhibitory scaling factor.
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.05 ms).
"""
@blox struct LIFExciNeuron(;
        name,
        namespace = nothing,
        g_L = 25 * 1e-6, # mS
        V_L = -70, # mV
        V_E = 0, # mV
        V_I = -70, # mV
        θ = -50, # mV
        V_reset = -55, # mV
        C = 0.5 * 1e-3, # mS / kHz 
        τ_AMPA = 2, # ms
        τ_GABA = 5, # ms
        τ_NMDA_decay = 100, # ms
        τ_NMDA_rise = 2, # ms
        t_refract = 2, # ms
        α = 0.5, # ms⁻¹
        g_AMPA = 0.05 * 1e-6, # mS
        g_AMPA_ext = 2.1 * 1e-6, # mS
        g_GABA = 1.3 * 1e-6, # mS
        g_NMDA = 0.165 * 1e-6, # mS  
        Mg = 1, # mM
        exci_scaling_factor = 1,
        inh_scaling_factor = 1,
        dtmax=0.05) <: AbstractExciNeuron
    
    @params( 
        g_L, 
        V_L, 
        V_E,
        V_I,
		V_reset,
        θ,
        C,
        τ_AMPA, 
        τ_GABA, 
        τ_NMDA_decay, 
        τ_NMDA_rise, 
        t_refract_duration=t_refract,
        t_refract_end=-Inf,
        g_AMPA = g_AMPA * exci_scaling_factor,
        g_AMPA_ext = g_AMPA_ext,
        g_GABA = g_GABA * inh_scaling_factor,
        g_NMDA = g_NMDA * exci_scaling_factor,
        α,
        Mg,
        is_refractory=0
    )
    @states(
        V=-52.0,
        S_AMPA=0.0,
        S_GABA=0.0,
        S_NMDA=0.0,
        x=0.0,
        S_AMPA_ext=0.0,
    )
    @inputs jcn=0.0
    @outputs V
    @equations begin
        D(V) = (1 - is_refractory) * (- g_L * (V - V_L) - S_AMPA_ext * g_AMPA_ext * (V - V_E) - S_GABA * g_GABA * (V - V_I) - S_AMPA * g_AMPA * (V - V_E) - jcn) / C
        D(S_AMPA) = - S_AMPA / τ_AMPA
        D(S_GABA) = - S_GABA / τ_GABA
        D(S_NMDA) = - S_NMDA / τ_NMDA_decay + α * x * (1 - S_NMDA)
        D(x) = - x / τ_NMDA_rise
        D(S_AMPA_ext) = - S_AMPA_ext / τ_AMPA
    end
end

# Paramater bounds for GUI
# C = [0.1, 100] μF
# E_syn = [1, 100] kΩ
# E_syn = [-10, 10] mV
# G_syn = [0.001, 0.01] μA/mV
# τ₁ = [1, 100] ms
# τ₂ = [1, 100] ms
# I_in = [-2.5, 2.5] μA 
# Eₘ = [-10, 10] mV
# Vᵣₑₛ = [-100, -55] mV
# θ = [0, 50] mV

"""
    QIFNeuron(; name, namespace, kwargs...)

Create a quadratic integrate-and-fire neuron.

The formal definition of this blox is:
```math
    \\frac{dV}{dt} = ((V - Eₘ)^2 / (Rₘ^2) + I_in + jcn) / C
    \\frac{dG}{dt} = (-1 / τ₂)G + z
    \\frac{dz}{dt} = (-1 / τ₁)z
```
where ``jcn`` is any input to the blox.

Keyword Arguments:
- C: Membrane capacitance (μF).
- Rₘ: membrane resistance (kΩ).
- E_syn: Synaptic reversal potential (mV).
- G_syn: Synaptic conductance (mS).
- τ₁: Timescale of decay of synaptic conductance (ms).
- τ₂: Timescale of decay of synaptic spike variable (ms).
- I_in: External current input (μA).
- Eₘ: Resting membrane potential (mV).
- Vᵣₑₛ: Post action potential (mV).
- θ: Threshold potential (mV).
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.05 ms).
"""
@blox struct QIFNeuron(;name, 
					   namespace=nothing,
					   C=1.0,
					   Rₘ = 10.0,
					   E_syn=0.0,
					   G_syn=0.002, 
					   τ₁=10.0,
					   τ₂=10.0,
					   I_in=0.0, 
					   Eₘ=0.0,
					   Vᵣₑₛ=-70.0,
					   θ=25.0,
                       dtmax=0.05) <: AbstractNeuron
    @params C Rₘ E_syn G_syn τ₁ τ₂ I_in Eₘ Vᵣₑₛ θ dtmax
    @states V=-70.0 G=0.0 z=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = ((V-Eₘ)^2/(Rₘ^2)+I_in+jcn)/C
		D(G) = (-1/τ₂)*G + z
	    D(z) = (-1/τ₁)*z
    end
    @discrete_events (V > θ) => (V=Vᵣₑₛ, z=G_syn)
end

# Paramater bounds for GUI
# α = [0.1, 1]
# η = [0, 1]
# a = [0.001, 0.5]
# b = [-0.01, 0.01]
# θ = [50, 250]
# vᵣ = [-250, -50]
# wⱼ = [0.001, 0.1]
# sⱼ = [0.5, 10]
# gₛ = [0.5, 10]
# eᵣ = [0.5, 10]
# τ = [1, 10]
# This is largely the Chen and Campbell Izhikevich implementation, with synaptic dynamics adjusted to reflect the LIF/QIF implementations above
"""
    IzhikevichNeuron(; name, namespace, kwargs...)

Create an Izhikevich neuron, largely following the implementation in Chen and Campbell (2022), with synaptic decay.

The formal definition of this blox is:

```math
\\frac{dV}{dt} = V(V - α) - w + η + jcn \\\\
\\frac{dw}{dt} = a(bV - w) \\\\
\\frac{dG}{dt} = -(1 / τ)G + z \\\\
\\frac{dz}{dt} = -(1 / τ)z
```

Arguments:
- α: The firing rate parameter (defaults to 0.6215).
- η: Intrinsic current (defaults to 0.12 mA).
- a: Timescale of the recovery variable (defaults to 0.0077 ms).
- b: Sensitivity of the recovery variable to fluctuations in the voltage (defaults to −0.0062).
- θ: Threshold voltage (defaults to 200.0 mV).
- vᵣ: Reset potential after a spike (defaults to -200.0 mV).
- wⱼ: The jump in the recovery variable after a spike (defaults to 0.0189).
- sⱼ: Reset value for the synaptic spike variable after a spike (defaults to 1.2308).
- gₛ: The synaptic conductance (defaults to 1.2308 mS).
- eᵣ: The synaptic reversal potential (defaults to 1.0 mV).
- τ: The synaptic decay time constant (defaults to 2.6 ms).
- dtmax: Maximum timestep allowed for adaptive ODE solvers (Default: 0.01 ms).

References:
1. Izhikevich, E. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569–1572.
2. Chen, L., & Campbell, S. A. (2022). Exact mean-field models for spiking neural networks with adaptation. Journal of Computational Neuroscience, 50(4), 445-469.
"""
@blox struct IzhikevichNeuron(;name,
							  namespace=nothing,
							  α=0.6215,
							  η=0.12,
							  a=0.0077,
							  b=-0.0062,
							  θ=200.0,
							  vᵣ=-200.0,
							  wⱼ=0.0189,
							  sⱼ=1.2308,
							  gₛ=1.2308,
							  eᵣ=1.0,
							  τ=2.6,
                              dtmax=0.01) <: AbstractNeuron
    @params(α, η, a, b, θ, vᵣ, wⱼ, sⱼ, gₛ, eᵣ, τ, dtmax)
    @states V=0.0 w=0.0 G=0.0 z=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = V*(V-α) - w + η + jcn
		D(w) = a*(b*V - w)
		D(G) = (-1/τ)*G + z
		D(z) = (-1/τ)*z
    end
    @discrete_events (V >= θ) => (V=vᵣ, w=w+wⱼ, z=sⱼ)
end


"""
    PINGNeuronExci(name, namespace, C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D)

Create an excitatory neuron from Borgers et al. (2008).
The formal definition of this blox is:

```math
\\frac{dV}{dt} = \\frac{1}{C}(-g_{Na} m_{*}^3*h*(V - V_{Na}) - g_K*n^4*(V - V_K) - g_L*(V - V_L) + I_{ext} + jcn) \\\\
\\m_{*} = \\frac{a_m(V)}{a_m(V) + b_m(V)} \\\\
\\frac{dn}{dt} = a_n(V)*(1 - n) - b_n(V)*n \\\\
\\frac{dh}{dt} = a_h(V)*(1 - h) - b_h(V)*h \\\\
\\frac{ds}{dt} = \\frac{1}{2}*(1 + \\text{tanh}(V/10))*(\\frac{1 - s}{\\tau_R} - \\frac{s}{\\tau_D})
```
where ``jcn`` is any input to the blox. Note that this is a modified Hodgkin-Huxley formalism with an additional synaptic accumulation term.
Synapses are added into the ``jcn`` term by connecting the postsynaptic neuron's voltage to the presynaptic neuron's output:
```math
jcn = w*s*(V_E - V)
```
where ``w`` is the weight of the synapse and ``V_E`` is the reversal potential of the excitatory synapse.

Inputs:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- C: Membrane capacitance (defaults to 1.0).
- g_Na: Sodium conductance (defaults to 100.0).
- V_Na: Sodium reversal potential (defaults to 50.0).
- g_K: Potassium conductance (defaults to 80.0).
- V_K: Potassium reversal potential (defaults to -100.0).
- g_L: Leak conductance (defaults to 0.1).
- V_L: Leak reversal potential (defaults to -67.0).
- I_ext: External current (defaults to 0.0).
- τ_R: Rise time of synaptic conductance (defaults to 0.2).
- τ_D: Decay time of synaptic conductance (defaults to 2.0).
"""
@blox struct PINGNeuronExci(;name,
                            namespace=nothing,
                            C=1.0,
                            g_Na=100.0,
                            V_Na=50.0,
                            g_K=80.0,
                            V_K=-100.0,
                            g_L=0.1,
                            V_L=-67.0,
                            I_ext=0.0,
                            τ_R=0.2,
                            τ_D=2.0) <: AbstractPINGNeuron
    @params C g_Na V_Na g_K V_K g_L V_L I_ext τ_R τ_D
    @states V=0.0 n=0.0 h=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        @setup begin
            a_m(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
            b_m(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
            a_n(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
            b_n(v) = 0.5*exp(-(v+57.0)/40.0)
            a_h(v) = 0.128*exp((v+50.0)/18.0)
            b_h(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
            m∞(v)  = a_m(v)/(a_m(v) + b_m(v))
        end
        D(V) = g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn
        D(n) = (a_n(V)*(1.0 - n) - b_n(V)*n)
        D(h) = (a_h(V)*(1.0 - h) - b_h(V)*h)
        D(s) = ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
    end
end

"""
    PINGNeuronInhib(name, namespace, C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D)

Create an inhibitory neuron from Borgers et al. (2008).
The formal definition of this blox is:

```math
\\frac{dV}{dt} = \\frac{1}{C}(-g_{Na}*m_{*}^3*h*(V - V_{Na}) - g_K*n^4*(V - V_K) - g_L*(V - V_L) + I_{ext} + jcn) \\\\
\\m_{*} = \\frac{a_m(V)}{a_m(V) + b_m(V)} \\\\
\\frac{dn}{dt} = a_n(V)*(1 - n) - b_n(V)*n \\\\
\\frac{dh}{dt} = a_h(V)*(1 - h) - b_h(V)*h \\\\
\\frac{ds}{dt} = \\frac{1}{2}*(1 + \\tanh(V/10))*(\\frac{1 - s}{\\tau_R} - \\frac{s}{\\tau_D})
```
where ``jcn`` is any input to the blox. Note that this is a modified Hodgkin-Huxley formalism with an additional synaptic accumulation term.
Synapses are added into the ``jcn`` term by connecting the postsynaptic neuron's voltage to the presynaptic neuron's output:
```math
jcn = w*s*(V_I - V)
```
where ``w`` is the weight of the synapse and ``V_I`` is the reversal potential of the inhibitory synapse.

Inputs:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- C: Membrane capacitance (defaults to 1.0).
- g_Na: Sodium conductance (defaults to 35.0).
- V_Na: Sodium reversal potential (defaults to 55.0).
- g_K: Potassium conductance (defaults to 9.0).
- V_K: Potassium reversal potential (defaults to -90.0).
- g_L: Leak conductance (defaults to 0.1).
- V_L: Leak reversal potential (defaults to -65.0).
- I_ext: External current (defaults to 0.0).
- τ_R: Rise time of synaptic conductance (defaults to 0.5).
- τ_D: Decay time of synaptic conductance (defaults to 10.0).
"""
@blox struct PINGNeuronInhib(;name,
                             namespace=nothing,
                             C=1.0,
                             g_Na=35.0,
                             V_Na=55.0,
                             g_K=9.0,
                             V_K=-90.0,
                             g_L=0.1,
                             V_L=-65.0,
                             I_ext=0.0,
                             τ_R=0.5,
                             τ_D=10.0) <: AbstractPINGNeuron
    @params C g_Na V_Na g_K V_K g_L V_L I_ext τ_R τ_D
    @states V=0.0 n=0.0 h=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        @setup begin
            a_m(v) = 0.1*(v+35.0)/(1.0 - exp(-(v+35.0)/10.0))
            b_m(v) = 4*exp(-(v+60.0)/18.0)
            a_n(v) = 0.05*(v+34.0)/(1.0 - exp(-(v+34.0)/10.0))
            b_n(v) = 0.625*exp(-(v+44.0)/80.0)
            a_h(v) = 0.35*exp(-(v+58.0)/20.0)
            b_h(v) = 5.0/(1.0 + exp(-(v+28.0)/10.0))

            m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        end
        D(V) = g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn
        D(n) = (a_n(V)*(1.0 - n) - b_n(V)*n)
        D(h) = (a_h(V)*(1.0 - h) - b_h(V)*h)
        D(s) = ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
    end 
end
