# First, create an abstract neuron that we'll extend to create the neurons for this tutorial.
abstract type AbstractPINGNeuron <: AbstractNeuronBlox end

"""
    PINGNeuronExci(name, namespace, C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D)

    Create an excitatory neuron from Borgers et al. (2008).
    The formal definition of this blox is:

```math
\\frac{dV}{dt} = \\frac{1}{C}(-g_{Na}*m_{\\infty}^3*h*(V - V_{Na}) - g_K*n^4*(V - V_K) - g_L*(V - V_L) + I_{ext} + jcn)
\\m_{\\infty} = \\frac{a_m(V)}{a_m(V) + b_m(V)}
\\frac{dn}{dt} = a_n(V)*(1 - n) - b_n(V)*n
\\frac{dh}{dt} = a_h(V)*(1 - h) - b_h(V)*h
\\frac{ds}{dt} = \\frac{1}{2}*(1 + \\tanh(V/10))*(\\frac{1 - s}{\\tau_R} - \\frac{s}{\\tau_D})
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
struct PINGNeuronExci <: AbstractPINGNeuron
    params
    system
    namespace

    function PINGNeuronExci(;name,
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
                             τ_D=2.0)
        p = paramscoping(C=C, g_Na=g_Na, V_Na=V_Na, g_K=g_K, V_K=V_K, g_L=g_L, V_L=V_L, I_ext=I_ext, τ_R=τ_R, τ_D=τ_D)
        C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D = p
        sts = @variables V(t)=0.0 n(t)=0.0 h(t)=0.0 s(t)=0.0 [output=true] jcn(t) [input=true]
        
        a_m(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
        b_m(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
        a_n(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
        b_n(v) = 0.5*exp(-(v+57.0)/40.0)
        a_h(v) = 0.128*exp((v+50.0)/18.0)
        b_h(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
        
        m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        eqs = [D(V) ~ g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn,
               D(n) ~ (a_n(V)*(1.0 - n) - b_n(V)*n),
               D(h) ~ (a_h(V)*(1.0 - h) - b_h(V)*h),
               D(s) ~ ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
        ]
        sys = ODESystem(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end

"""
    PINGNeuronInhib(name, namespace, C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D)

    Create an inhibitory neuron from Borgers et al. (2008).
    The formal definition of this blox is:

```math
\\frac{dV}{dt} = \\frac{1}{C}(-g_{Na}*m_{\\infty}^3*h*(V - V_{Na}) - g_K*n^4*(V - V_K) - g_L*(V - V_L) + I_{ext} + jcn)
\\m_{\\infty} = \\frac{a_m(V)}{a_m(V) + b_m(V)}
\\frac{dn}{dt} = a_n(V)*(1 - n) - b_n(V)*n
\\frac{dh}{dt} = a_h(V)*(1 - h) - b_h(V)*h
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
struct PINGNeuronInhib <: AbstractPINGNeuron
    params
    system
    namespace

    function PINGNeuronInhib(;name,
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
                             τ_D=10.0)
        p = paramscoping(C=C, g_Na=g_Na, V_Na=V_Na, g_K=g_K, V_K=V_K, g_L=g_L, V_L=V_L, I_ext=I_ext, τ_R=τ_R, τ_D=τ_D)
        C, g_Na, V_Na, g_K, V_K, g_L, V_L, I_ext, τ_R, τ_D = p
        sts = @variables V(t)=0.0 n(t)=0.0 h(t)=0.0 s(t)=0.0 [output=true] jcn(t) [input=true]

        a_m(v) = 0.1*(v+35.0)/(1.0 - exp(-(v+35.0)/10.0))
        b_m(v) = 4*exp(-(v+60.0)/18.0)
        a_n(v) = 0.05*(v+34.0)/(1.0 - exp(-(v+34.0)/10.0))
        b_n(v) = 0.625*exp(-(v+44.0)/80.0)
        a_h(v) = 0.35*exp(-(v+58.0)/20.0)
        b_h(v) = 5.0/(1.0 + exp(-(v+28.0)/10.0))

        m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        eqs = [D(V) ~ g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn,
               D(n) ~ (a_n(V)*(1.0 - n) - b_n(V)*n),
               D(h) ~ (a_h(V)*(1.0 - h) - b_h(V)*h),
               D(s) ~ ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
        ]
        sys = ODESystem(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
        end
end

