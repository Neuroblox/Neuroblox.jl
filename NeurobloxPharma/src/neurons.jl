"""
    HHNeuronExci(; I_bg = 0)

Excitatory neuron using the Hodgkin-Huxley formalism [1]. 

Equations were based on the supplementary material of [2] and the threshold values α and β were based on [3]:
```math
\\frac{dV}{dt} = - G_Na m^3 h (V-E_\\text{Na}) - G_K n^4 (V-E_\\text{K}) - G_L (V-E_L) + I_\\text{bg} + I_\\text{syn} + \\text{jcn} \\\\
\\frac{dn}{dt} = ϕ*(α_n(V)*(1-n)-β_n(V)*n) \\\\
\\frac{dm}{dt} = ϕ*(α_m(V)*(1-m)-β_m(V)*m) \\\\
\\frac{dh}{dt} = ϕ*(α_h(V)*(1-h)-β_h(V)*h) \\\\
α_n(V) = 0.01 \\frac{V+34}{1- e ^{-\\frac{V+34}{10}}} \\\\
β_n(V) = 0.125 e^{-\\frac{V+44}{80}} \\\\
α_m(V) = 0.1 \\frac{V+30}{1 - e^{-\\frac{V+30}{10}}}
β_m(V) = 4 e^{-\\frac{V+55}{18}}
α_h(V) = 0.07 e^{-\\frac{V+44}{20}}
β_h(V) = \\frac{1}{1 + e^{-\\frac{V+14}{10}}}
```

Model parameters : 
- Based on [3] : 
    - `G_Na = 52` [mV, Na channel conductance]
	- `G_K = 20`  [mV, K channel conductance] 
    - `E_Na = 55` [mV, Na channel reversal potential]
    - `E_K = -90` [mV, K channel reversal potential]
    - `E_L = -60` [mV, leak reversal potential]
    - `ϕ = 5` [temperature effect on timescale]

Arguments : 
- `I_bg` [μA, background current] 

References:
1. AL Hodgkin, AF Huxley, A quantitative description of membrane current and its application to conduction and excitation in nerve. The J. Physiol., 117, 500–544, 1952.
2. Pathak, A., Brincat, S.L., Organtzidis, H. et al., Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun. 2025.
3. XJ Wang, Ionic basis for intrinsic 40 Hz neuronal oscillations. NeuroReport 5, 221–224, 1993.
"""
@blox struct HHNeuronExci(;name, namespace=nothing, E_syn=0.0, G_syn=3, I_bg=0.0, τ=5.0) <: AbstractExciNeuron
    @params(
        E_syn=E_syn, 
		G_Na = 52, 
		G_K  = 20,
		G_L = 0.1, 
		E_Na = 55,
		E_K = -90,
		E_L = -60,
		G_syn = G_syn, 
		V_shift = 10,
		V_range = 35,
		τ = τ,
		I_bg=I_bg,
		kₛₜₚ = 0.5,
        spikes = 0,
		spk_const = 1.127
    )
    @states(
        V=-65.0,
        n=0.32,
        m=0.05,
        h=0.59,
        spikes_cumulative=0.0,
        spikes_window=0.0
    )
    @inputs I_syn=0.0 I_in=0.0 I_asc=0.0 jcn=0.0
    @equations begin
        @setup begin
            αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
	        βₙ(v) = 0.125*exp(-(v+44)/80)
	        αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
	        βₘ(v) = 4*exp(-(v+55)/18)
	        αₕ(v) = 0.07*exp(-(v+44)/20)
	        βₕ(v) = 1/(1+exp(-(v+14)/10))
	        ϕ = 5
            # These are not real synaptic paramters, they're ficticious and just used for spike counting in the
            # neuron.
            _V_shift = 10.0
            _V_range = 35.0
            _G_syn = 3.0
	        _G_asymp = (_G_syn/(1 + exp(-4.394*((V - _V_shift)/_V_range))))
        end
        D(V) = -G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn
		D(n) = ϕ*(αₙ(V)*(1-n)-βₙ(V)*n) 
		D(m) = ϕ*(αₘ(V)*(1-m)-βₘ(V)*m)
		D(h) = ϕ*(αₕ(V)*(1-h)-βₕ(V)*h)
        D(spikes_cumulative) = spk_const*_G_asymp
        D(spikes_window)     = spk_const*_G_asymp
    end
end
default_synapse_type(::HHNeuronExci, ::AbstractNeuron) = Glu_AMPA_Synapse

NeurobloxBase.has_t_block_event(::Type{HHNeuronExci}) = true
NeurobloxBase.is_t_block_event_time(::Type{HHNeuronExci}, key, t) = key == :t_block_late
NeurobloxBase.t_block_event_requires_inputs(::Type{HHNeuronExci}) = false
function NeurobloxBase.apply_t_block_event!(vstates, _, s::Subsystem{HHNeuronExci}, _, _)
    vstates.spikes_window[] = 0.0
end

"""
    Glu_AMPA_Synapse(; E_syn=0,  G_syn=3, V_shift=10, V_range=35, τ₁=0.1, τ₂=5, g=1)

An AMPA receptor model activated by glutamate. Equations and default parameter values are based on [1, 2].

```math
\\frac{dG}{dt} = -\\frac{G}{τ_2} + z \\\\
\\frac{dz}{dt} = -\\frac{z}{τ_1} + \\frac{G_\\text{syn}}{1 + e^{-4.394(\\frac{V-V_\\text{shift}}{V_\\text{range}})}}
```

Arguments : 
- `E_syn` [mV, reversal potential]
- `G_syn` [mV, receptor conductance]
- `V_shift` [mV, transmitter threshold]
- `V_range` [mV, transmitter sensitivity]
- `τ₁` [ms, decay timescale for receptor conductance]
- `τ₂` [ms, decay timescale for receptor conductance]
- `g` [conductance gain]

References : 
1. C Koch, I Segev, TJ Sejnowski, TA Poggio, eds. (1998), Methods in Neuronal Modeling: From Ions to Networks, Computational
Neuroscience Series. (A Bradford Book, Cambridge, MA, USA), 2 edition.
2. SR Wicks, CJ Roehrig, CH Rankin (1996), A Dynamic Network Simulation of the Nematode Tap Withdrawal Circuit: Predictions
Concerning Synaptic Function Using Behavioral Criteria. The J. Neurosci. 16, 4017–4031.
"""
@blox struct Glu_AMPA_Synapse(;name, namespace=nothing,
                              E_syn=0.0, G_syn=3.0, V_shift=10.0, V_range=35.0, τ₁=0.1, τ₂=5.0, g=1.0) <: AbstractReceptor
    @params E_syn G_syn V_shift V_range τ₁ τ₂ g
    @states G=0.0 z=0.0
    @inputs V=0.0
    @outputs G
    @equations begin
        D(G) = -G/τ₂ + z
        D(z) = -z/τ₁ + (G_syn/(1 + exp(-4.394*((V-V_shift)/V_range))))
    end
end

# Glu_AMPA_Synapse inherits from the presynaptic neuron
function Glu_AMPA_Synapse(src::AbstractNeuron, ::AbstractNeuron; name=:glu_ampa, namespace=full_namespaced_nameof(src), g=1.0, kwargs...)
    (; E_syn, G_syn, V_shift, V_range, τ) = merge(src.param_vals, NamedTuple(kwargs))
    Glu_AMPA_Synapse(; name, namespace, E_syn, G_syn, V_shift, V_range, τ₂=τ, g)
end

"""
    Glu_AMPA_STA_Synapse(; E_syn=0,  G_syn=3, V_shift=10, V_range=35, τ₃=2000, τ₄=0.1, kₛₜₚ=0.5, g=1)

An AMPA receptor model activated by glutamate, which also includes short-term augmentation dynamics. Equations and default parameter values are based on [1, 2].

```math
\\frac{d G_\\text{stp}}{dt} = -\\frac{G_\\text{stp}}{τ_3} + (k_\\text{stp} - G_\\text{stp}) \\frac{z_\\text{stp}}{5} \\\\
\\frac{d z_\\text{stp}}{dt} = -\\frac{z_\\text{stp}}{τ_4} + \\frac{G_\\text{syn}}{1 + e^{-4.394(\\frac{V-V_\\text{shift}}{V_\\text{range}})}}
```

Arguments : 
- `E_syn` [mV, reversal potential]
- `G_syn` [mV, receptor conductance]
- `V_shift` [mV, transmitter threshold]
- `V_range` [mV, transmitter sensitivity]
- `τ₁` [ms, decay timescale for receptor conductance]
- `τ₃` [ms, decay timescale for receptor conductance]
- `kₛₜₚ` [mV, asymptotic upper bound of Gₛₜₚ]
- `g` [conductance gain]

References : 
1. C Koch, I Segev, TJ Sejnowski, TA Poggio, eds. (1998), Methods in Neuronal Modeling: From Ions to Networks, Computational
Neuroscience Series. (A Bradford Book, Cambridge, MA, USA), 2 edition.
2. SR Wicks, CJ Roehrig, CH Rankin (1996), A Dynamic Network Simulation of the Nematode Tap Withdrawal Circuit: Predictions
Concerning Synaptic Function Using Behavioral Criteria. The J. Neurosci. 16, 4017–4031.
"""
@blox struct Glu_AMPA_STA_Synapse(;name, namespace=nothing,
                                E_syn=0.0, G_syn=3.0, V_shift=10.0, V_range=35.0,
                                τ₃=2000.0, τ₁=0.1, kₛₜₚ=0.5, g=1.0, kwargs...) <: AbstractReceptor
    @params E_syn G_syn V_shift V_range τ₃ τ₁ kₛₜₚ g
    @states Gₛₜₚ=0.0 zₛₜₚ=0.0 
    @inputs V=0.0
    @outputs Gₛₜₚ
    @equations begin
        D(Gₛₜₚ) = -Gₛₜₚ/τ₃ + (zₛₜₚ/5)*(kₛₜₚ-Gₛₜₚ)
        D(zₛₜₚ) = -zₛₜₚ/τ₁ + (G_syn/(1 + exp(-4.394*((V-V_shift)/V_range))))
    end
end

# Glu_AMPA_STA_Synapse inherits from the postsynaptic neuron
function Glu_AMPA_STA_Synapse(::HHNeuronExci, dst::HHNeuronExci; name=:glu_ampa_sta, namespace=full_namespaced_nameof(dst), g=1.0, kwargs...)
    (; E_syn, G_syn, V_shift, V_range, kₛₜₚ, τ) = merge(dst.param_vals, NamedTuple(kwargs))
    Glu_AMPA_STA_Synapse(; name, namespace, E_syn, τ₂=τ, kₛₜₚ, g)
end

function get_synapse(src, dst; synapse_type=default_synapse_type(src, dst), synapse_kwargs=(;), synapse=nothing, kwargs...)
    if !isnothing(synapse)
        synapse
    else
        synapse_type(src, dst; synapse_kwargs...)
    end
end

"""
    HHNeuronInhib(; I_bg = 0)

Inhibitory neuron using the Hodgkin-Huxley formalism [1]. 

Equations were based on the supplementary material of [2] and the threshold values α and β were based on [3]:
```math
\\frac{dV}{dt} = - G_Na m^3 h (V-E_\\text{Na}) - G_K n^4 (V-E_\\text{K}) - G_L (V-E_L) + I_\\text{bg} + I_\\text{syn} + \\text{jcn} \\\\
\\frac{dn}{dt} = ϕ*(α_n(V)*(1-n)-β_n(V)*n) \\\\
\\frac{dm}{dt} = ϕ*(α_m(V)*(1-m)-β_m(V)*m) \\\\
\\frac{dh}{dt} = ϕ*(α_h(V)*(1-h)-β_h(V)*h) \\\\
α_n(V) = 0.01 \\frac{V+34}{1- e ^{-\\frac{V+34}{10}}} \\\\
β_n(V) = 0.125 e^{-\\frac{V+44}{80}} \\\\
α_m(V) = 0.1 \\frac{V+30}{1 - e^{-\\frac{V+30}{10}}}
β_m(V) = 4 e^{-\\frac{V+55}{18}}
α_h(V) = 0.07 e^{-\\frac{V+44}{20}}
β_h(V) = \\frac{1}{1 + e^{-\\frac{V+14}{10}}}
```

Model parameters based on [3] : 
    - `G_Na = 52` [mV, Na channel conductance]
	- `G_K = 20`  [mV, K channel conductance] 
    - `E_Na = 55` [mV, Na channel reversal potential]
    - `E_K = -90` [mV, K channel reversal potential]
    - `E_L = -60` [mV, leak reversal potential]
    - `ϕ = 5` [temperature effect on timescale]

Arguments : 
- `I_bg` [μA, background current] 

References:
1. AL Hodgkin, AF Huxley, A quantitative description of membrane current and its application to conduction and excitation in nerve. The J. Physiol. 117, 500–544, 1952.
2. Pathak, A., Brincat, S.L., Organtzidis, H. et al., Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun. 2025.
3. XJ Wang, Pacemaker Neurons for the Theta Rhythm and Their Synchronization in the Septohippocampal Reciprocal Loop. J. Neurophysiol. 87, 889–900, 2002.
"""
@blox struct HHNeuronInhib(;name, namespace=nothing,
                           E_syn=-70.0,
                           G_syn=11.5,
                           I_bg=0.0,
                           τ=70.0) <: AbstractInhNeuron
    @params(
        E_syn=E_syn,
		G_Na = 52, 
		G_K  = 20, 
		G_L = 0.1, 
		E_Na = 55, 
		E_K = -90,
		E_L = -60, 
		G_syn = G_syn, 
		V_shift = 0, 
		V_range = 35, 
		τ = τ,
		I_bg=I_bg, 
    )
    @states(
        V=-65.00,
		n=0.32, 
		m=0.05, 
		h=0.59, 
    )
    @inputs I_syn=0.0 I_in=0.0 I_asc=0.0 jcn=0.0
    @equations begin
        @setup begin
	        αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
	        βₙ(v) = 0.125*exp(-(v+48)/80)
            αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
	        βₘ(v) = 4*exp(-(v+58)/18)
            αₕ(v) = 0.07*exp(-(v+51)/20)
	        βₕ(v) = 1/(1+exp(-(v+21)/10))   	
	        ϕ = 5
        end
		D(V) = -G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn
		D(n) =  ϕ*(αₙ(V)*(1-n)-βₙ(V)*n) 
		D(m) =  ϕ*(αₘ(V)*(1-m)-βₘ(V)*m)
		D(h) =  ϕ*(αₕ(V)*(1-h)-βₕ(V)*h)
    end
end
default_synapse_type(::HHNeuronInhib, ::AbstractNeuron) = GABA_A_Synapse

"""
    GABA_A_Synapse(; E_syn=-70, G_syn=11.5, τ₁=0.1, τ₂=70, g=1, V_shift=0, V_range=35)

A GABA A receptor model using the same type of damped oscillator dynamics as [`Glu_AMPA_Synapse`](@ref). Equations and default parameter values are based on [1, 2].

```math
\\frac{dG}{dt} = -\\frac{G}{τ_2} + z \\\\
\\frac{dz}{dt} = -\\frac{z}{τ_1} + \\frac{G_\\text{syn}}{1 + e^{-4.394(\\frac{V-V_\\text{shift}}{V_\\text{range}})}}
```

Arguments : 
- `E_syn` [mV, reversal potential]
- `G_syn` [mV, receptor conductance]
- `V_shift` [mV, transmitter threshold]
- `V_range` [mV, transmitter sensitivity]
- `τ₁` [ms, decay timescale for receptor conductance]
- `τ₂` [ms, decay timescale for receptor conductance]
- `g` [conductance gain]

References : 
1. C Koch, I Segev, TJ Sejnowski, TA Poggio, eds. (1998), Methods in Neuronal Modeling: From Ions to Networks, Computational
Neuroscience Series. (A Bradford Book, Cambridge, MA, USA), 2 edition.
2. SR Wicks, CJ Roehrig, CH Rankin (1996), A Dynamic Network Simulation of the Nematode Tap Withdrawal Circuit: Predictions
Concerning Synaptic Function Using Behavioral Criteria. The J. Neurosci. 16, 4017–4031.
"""
@blox struct GABA_A_Synapse(;name, namespace=nothing,
                            E_syn=-70.0,  G_syn=11.5, τ₁=0.1, τ₂=70.0, g=1.0, 
                            V_shift=0.0, V_range=35) <: AbstractReceptor
    @params E_syn G_syn τ₁ τ₂ V_shift V_range g 
    @states G=0.0 z=0.0
    @inputs V=0.0
    @outputs G
    @equations begin
        D(G) = -G/τ₂ + z
        D(z) = -z/τ₁ + (G_syn/(1 + exp(-4.394*((V-V_shift)/V_range))))
    end
end

# GABA_A_Synapse inherits from the presynaptic neuron
function GABA_A_Synapse(n::HHNeuronInhib, ::AbstractNeuron; name=:gaba_a, namespace=full_namespaced_nameof(n), g=1.0, kwargs...)
    (; E_syn, G_syn, τ, V_shift, V_range) = merge(n.param_vals, NamedTuple(kwargs))
    GABA_A_Synapse(; name, namespace, E_syn, G_syn, τ₂=τ, V_shift, V_range, g)
end

# FSI stuff, TODO: receptor blox for FSI neurons?
@blox struct HHNeuronFSI(
    ;name,
    namespace=nothing,
    E_syn=-80.0,
    I_bg=6.2,
    τ=11.0,
	τₛ=6.5,
    Cₘ=1.0,
	σ=1.2,
	a=4.0,
	b=10.0,
	T=37.0
    ) <: AbstractInhNeuron
    
    @params(
        E_syn=E_syn,
		G_Na = 112.5, 
		G_K  = 225.0, 
		G_L = 0.25, 
		G_D = 6.0,
		E_Na = 50.0, 
		E_K = -90.0, 
		E_L = -70.0, 
		I_bg=I_bg,
        Cₘ = Cₘ,
		σ = σ,
		a = a,
		b = b,
		T = T,
        τ = τ,
        τₛ = τₛ,
    )
    @states(
        V=-70.00, 
		n=0.032, 
		h=0.059, 
		mD=0.05,
		hD=0.059,
        G =0.0,
        Gₛ=0.0,
    )
    @inputs(
        I_syn=0.0,
        I_gap=0.0,
        I_in =0.0,
        I_asc=0.0,
    )
    @outputs G Gₛ
    @equations begin
        @setup begin
		    n_inf(v) = 1/(1+exp(-(v+12.4)/6.8))
	        τₙ(v) = (0.087+11.4/(1+exp((v+14.6)/8.6)))*(0.087+11.4/(1+exp(-(v-1.3)/18.7)))
	        m_inf(v) = 1/(1+exp(-(v+24)/11.5))
     	    h_inf(v) = 1/(1+exp((v+58.3)/6.7))
	        τₕ(v) = 0.5 + 14/(1+exp((v+60)/12))
		    mD_inf(v) = 1/(1+exp(-(v+50)/20))
		    τₘD(v) = 2
		    hD_inf(v) = 1/(1+exp((v+70)/6))
		    τₕD(v) = 150
		    G_asymp(v,a,b) = a*(1+tanh(v/b))
        end
		D(V) =(1/Cₘ)*(-G_Na*m_inf(V)^3*h*(V-E_Na)-G_K*n^2*(V-E_K)-G_L*(V-E_L)-G_D*mD^3*hD*(V-E_K)+I_bg+I_syn+I_gap+I_asc+I_in+σ) 
		D(n) =(n_inf(V)-n)/τₙ(V)
		D(h) =(h_inf(V)-h)/τₕ(V)
		D(mD)=(mD_inf(V)-mD)/τₘD(V)
		D(hD)=(hD_inf(V)-hD)/τₕD(V)
		D(G) =(-1/τ)*G + G_asymp(V,a,b)*(1-G)
		D(Gₛ)=(-1/τₛ)*Gₛ + G_asymp(V,a,b)*(1-Gₛ)
    end
end

"""
    BaxterSensoryNeuron(; C_m=0.13, E_Na=60.0, E_Ca=60.0, E_K=-70.0, E_KCa=-65.0, E_L=-40.0, ...)

Aplysia sensory neuron conductance-based model with kinase-dependent parameter “corner” selection.
This neuron model implements the Baxter et al. (1999) sensory neuron equations, including:
- a membrane potential equation with multiple ionic currents (Na, Ca-N, Ca-L, K-A, K-V, K,S-V, K,S-I, K,Ca-F, K,Ca-S, leak),
- a Ca pool with Ca-dependent attenuation of Ca currents,
- fast/slow Ca-driven enhancers for Ca-activated K currents, and
- voltage-dependent gating variables with the paper’s steady-state and time-constant forms.

The model is designed to be paired with the `HTR5` mode selector blox:
binary inputs `PKA` and `PKC` (ideally 0/1) are used to bilinearly interpolate (or exactly select,
when binary) between the four Table-2 parameter corners:
Control (00), PKA-only (10), PKC-only (01), and 5-HT (11).

Core membrane equation (paper Appendix A1 form):
```math
C_m \\frac{dV}{dt} = I_{inj} - \\sum I_{ion}
```

Arguments:
- `C_m` [nF]: Membrane capacitance.
- `E_Na`, `E_Ca`, `E_K`, `E_KCa`, `E_L` [mV]: Reversal potentials.
- Table-2 corner parameters: conductances and selected time-constant extrema that depend on (`PKA`, `PKC`).
- Table-1 fixed maximal conductances: `gNa`, `gCaN`, `gKA`, `gKCaF`, `gL`.
- Ca pool and Ca-dependent regulation parameters: `K_Ca`, `τ_Ca`, `τ_gbr_F`, `τ_gbr_S`, etc.
- Gating parameters: (`hA`, `sA`, `tAmax`, `tAmin`, ...) for each channel as listed in the code.

Inputs:
- `jcn` [nA]: Generic Neuroblox synaptic / network current input.
- `I_stim` [nA]: External injected stimulus current.
- `I_bias_extra` [nA]: Optional extra bias current.
- `PKA` [-]: Kinase flag (0/1 recommended).
- `PKC` [-]: Kinase flag (0/1 recommended).

Outputs:
- `V` [mV]: Membrane potential.

References:
1. Baxter, D. A., Byrne, J. H. (1999). Serotonergic modulation of two potassium currents in Aplysia sensory neurons. *Journal of Neurophysiology*, 82(6), 2914–2926. (doi:10.1152/jn.1999.82.6.2914)
"""
@blox struct BaxterSensoryNeuron(;
    name,
    namespace = nothing,

    # ------------------------------------------------------------
    # Passive membrane property (Appendix: Cm = 0.13 nF)
    # ------------------------------------------------------------
    C_m = 0.13,               # nF

    # ------------------------------------------------------------
    # Reversal potentials (Table 1)
    # ------------------------------------------------------------
    E_Na  =  60.0,            # mV
    E_Ca  =  60.0,            # mV
    E_K   = -70.0,            # mV
    E_KCa = -65.0,            # mV
    E_L   = -40.0,            # mV


    # ----------------------------
    # Table 2 corner values
    # Index convention:
    #   _00 = Control  (PKA,PKC)=(0,0)
    #   _10 = PKA-only (1,0)
    #   _01 = PKC-only (0,1)
    #   _11 = 5-HT     (1,1)
    # ----------------------------
    gKS_V_00 = 0.54,  gKS_V_10 = 0.25,  gKS_V_01 = 0.34,  gKS_V_11 = 0.25,   # IK,S-V gmax, mS
    gKS_I_00 = 0.012, gKS_I_10 = 0.006, gKS_I_01 = 0.008, gKS_I_11 = 0.006,  # IK,S-I gmax
    gKV_00   = 4.5,   gKV_10   = 4.5,   gKV_01   = 2.3,   gKV_11   = 2.3,    # IK-V gmax

    tAmax_KV_00 = 32.5,  tAmax_KV_10 = 47.2,  tAmax_KV_01 = 49.8,  tAmax_KV_11 = 55.5,   # IK-V τA(max), ms
    tAmin_KV_00 = 0.5,   tAmin_KV_10 = 0.8,   tAmin_KV_01 = 0.88,  tAmin_KV_11 = 1.0,    # IK-V τA(min), ms
    tBmax_KV_00 = 1600.0,tBmax_KV_10 = 8200.0,tBmax_KV_01 = 8200.0,tBmax_KV_11 = 8200.0, # IK-V τB(max), ms
    tBmin_KV_00 = 5.0,   tBmin_KV_10 = 50.0,  tBmin_KV_01 = 50.0,  tBmin_KV_11 = 50.0,   # IK-V τB(min), ms

    gCaL_00  = 0.08,  gCaL_10  = 0.2,   gCaL_01  = 0.2,   gCaL_11  = 0.2,    # ICa-L gmax, mS
    gKCaS_00 = 0.158, gKCaS_10 = 0.036, gKCaS_01 = 0.08,  gKCaS_11 = 0.036,  # IK,Ca-S gmax, mS
    I_bias_00 = 0.0,   I_bias_10 = -0.11, I_bias_01 = -0.09, I_bias_11 = -0.11, # Table 2 IStim(bias)

    # ----------------------------
    # Table 1 fixed maximal conductances (units consistent with nA when V is mV)
    # These are NOT modulated in Table 2.
    # ----------------------------
    gNa   = 2.5,
    gCaN  = 0.16,
    gKA   = 0.65,
    gKCaF = 0.02,
    gL    = 0.018,

    # ----------------------------
    # Ca pool + Ca-dependent regulation (Appendix A5–A8)
    # ----------------------------
    K_Ca        = 18.0,
    τ_Ca        = 200.0,   # ms
    τ_gbr_F     = 10.0,    # ms
    τ_gbr_S     = 3000.0,  # ms
    K_CaN_inact = 60.0,
    K_CaL_inact = 75.0,
    b_Ca_inact  = 17.0,

    # ----------------------------
    # INa gating (Table 1)
    # ----------------------------
    hA_Na   = -10.1, sA_Na   = -9.8,
    tAmax_Na = 1.5,  tAmin_Na = 0.45, htA_Na = -0.7, stA_Na = 1.85,
    hB_Na   = -19.5, sB_Na   =  9.2,
    tBmax_Na = 15.0, tBmin_Na = 2.4,  htB_Na =  2.8, stB_Na = 3.5,

    # ----------------------------
    # ICa-N gating (Table 1; τA and τB use 2-exponential form)
    # ----------------------------
    hA_CaN   = 0.15, sA_CaN   = -5.0,
    tAmax_CaN = 10.0, tAmin_CaN = 0.05, htA1_CaN =  3.0, stA1_CaN = 18.4, htA2_CaN = -4.0,  stA2_CaN = -4.6,
    hB_CaN   = -15.0, sB_CaN   = 11.0,
    tBmax_CaN = 620.0, tBmin_CaN = 3.72, htB1_CaN = -18.0, stB1_CaN = 18.0, htB2_CaN = -20.0, stB2_CaN = -15.0,

    # ----------------------------
    # ICa-L gating (Table 1; τA and τB use 2-exponential form)
    # ----------------------------
    hA_CaL   = -7.4, sA_CaL   = -10.0,
    tAmax_CaL = 17.5, tAmin_CaL = 0.14, htA1_CaL =  2.2, stA1_CaL = 18.5, htA2_CaL = -15.0, stA2_CaL = -12.0,
    hB_CaL   = -33.0, sB_CaL   = 49.5,
    tBmax_CaL = 20500.0, tBmin_CaL = 20.5, htB1_CaL = -125.0, stB1_CaL = 40.0, htB2_CaL = -70.0, stB2_CaL = -115.0,

    # ----------------------------
    # IK-A gating (Table 1; τA uses 2-exponential form)
    # ----------------------------
    hA_KA   = -45.0, sA_KA   = -15.0,
    tAmax_KA = 17.0, tAmin_KA = 0.34, htA1_KA = -20.0, stA1_KA = 20.0, htA2_KA = -110.0, stA2_KA = -24.0,
    hB_KA   = -70.0, sB_KA   =  5.0,
    tBmax_KA = 224.0, tBmin_KA = 0.1, htB_KA = -54.4, stB_KA = 30.8,

    # ----------------------------
    # IK-V gating (Table 1; τB uses 2-exponential; special B_inf in Appendix)
    # ----------------------------
    hA_KV  = 15.3, sA_KV  = -8.4,
    htA_KV = 15.0, stA_KV =  8.2,
    hB_KV  =  7.9, sB_KV  =  1.5,
    htB1_KV = 5.8, stB1_KV = 3.3, htB2_KV = 6.3, stB2_KV = -1.5,

    # ----------------------------
    # IK,S-V gating (Table 1; activation only; τA uses 2-exponential form)
    # ----------------------------
    hA_KS_V   = 20.0, sA_KS_V   = -15.0,
    tAmax_KS_V = 255.0, tAmin_KS_V = 61.2, htA1_KS_V = -15.0, stA1_KS_V = 10.0, htA2_KS_V = -46.0, stA2_KS_V = -6.5,

    # ----------------------------
    # IK,Ca-F gating (Table 1; activation only; τA constant)
    # ----------------------------
    hA_KCaF = 23.5, sA_KCaF = -10.5,
    τA_KCaF = 1.0,

    # ----------------------------
    # IK,S-I instantaneous component (Table 1)
    # ----------------------------
    hA_KS_I = 91.2, sA_KS_I = -84.8
) <: AbstractExciNeuron

    @params(
        C_m, E_Na, E_Ca, E_K, E_KCa, E_L,

        # Table 2 corners
        gKS_V_00, gKS_V_10, gKS_V_01, gKS_V_11,
        gKS_I_00, gKS_I_10, gKS_I_01, gKS_I_11,
        gKV_00,   gKV_10,   gKV_01,   gKV_11,
        tAmax_KV_00, tAmax_KV_10, tAmax_KV_01, tAmax_KV_11,
        tAmin_KV_00, tAmin_KV_10, tAmin_KV_01, tAmin_KV_11,
        tBmax_KV_00, tBmax_KV_10, tBmax_KV_01, tBmax_KV_11,
        tBmin_KV_00, tBmin_KV_10, tBmin_KV_01, tBmin_KV_11,
        gCaL_00, gCaL_10, gCaL_01, gCaL_11,
        gKCaS_00, gKCaS_10, gKCaS_01, gKCaS_11,
        I_bias_00, I_bias_10, I_bias_01, I_bias_11,

        # Fixed gmax (Table 1)
        gNa, gCaN, gKA, gKCaF, gL,

        # Ca pool + regulation (Appendix)
        K_Ca, τ_Ca, τ_gbr_F, τ_gbr_S, K_CaN_inact, K_CaL_inact, b_Ca_inact,

        # Gating params (Table 1)
        hA_Na, sA_Na, tAmax_Na, tAmin_Na, htA_Na, stA_Na, hB_Na, sB_Na, tBmax_Na, tBmin_Na, htB_Na, stB_Na,
        hA_CaN, sA_CaN, tAmax_CaN, tAmin_CaN, htA1_CaN, stA1_CaN, htA2_CaN, stA2_CaN, hB_CaN, sB_CaN,
        tBmax_CaN, tBmin_CaN, htB1_CaN, stB1_CaN, htB2_CaN, stB2_CaN,
        hA_CaL, sA_CaL, tAmax_CaL, tAmin_CaL, htA1_CaL, stA1_CaL, htA2_CaL, stA2_CaL, hB_CaL, sB_CaL,
        tBmax_CaL, tBmin_CaL, htB1_CaL, stB1_CaL, htB2_CaL, stB2_CaL,
        hA_KA, sA_KA, tAmax_KA, tAmin_KA, htA1_KA, stA1_KA, htA2_KA, stA2_KA, hB_KA, sB_KA, tBmax_KA, tBmin_KA, htB_KA, stB_KA,
        hA_KV, sA_KV, htA_KV, stA_KV, hB_KV, sB_KV, htB1_KV, stB1_KV, htB2_KV, stB2_KV,
        hA_KS_V, sA_KS_V, tAmax_KS_V, tAmin_KS_V, htA1_KS_V, stA1_KS_V, htA2_KS_V, stA2_KS_V,
        hA_KCaF, sA_KCaF, τA_KCaF,
        hA_KS_I, sA_KS_I
    )

    @states(
        # Membrane potential and Ca pool (Appendix A1, A5)
        V     = -50.0,  # mV
        Ca    = 0.0,    # Ca pool (dimensionless in paper)
        gbr_F = 0.0,    # fast Ca-driven enhancer (Appendix A7)
        gbr_S = 0.0,    # slow Ca-driven enhancer (Appendix A7)

        # Gating variables (Appendix A3/A4)
        A_Na  = 0.0, B_Na  = 1.0,
        A_CaN = 0.0, B_CaN = 1.0,
        A_CaL = 0.0, B_CaL = 1.0,
        A_KA  = 0.0, B_KA  = 1.0,
        A_KV  = 0.0, B_KV  = 1.0,
        A_KS_V = 0.0,
        A_KCaF = 0.0
    )

    @inputs(
        # Neuroblox convention: "jcn" is where generic Neuron->Neuron connections accumulate.
        jcn          = 0.0,  # nA

        # External injected current (nA), plus optional extra bias (nA).
        I_stim       = 0.0,
        I_bias_extra = 0.0,

        # Kinase flags (ideally binary 0/1, driven by 5HTModeSelector).
        PKA = 0.0,
        PKC = 0.0
    )

    @outputs V

    @equations begin
        @setup begin
            # ----------------------------
            # Appendix A4: steady-state gating
            # ----------------------------
            A_inf(Vm, hA, sA) = 1.0 / (1.0 + exp((Vm - hA) / sA))

            # Appendix note: only IK-V uses pB=2 and Bmin=0.07; others use pB=1, Bmin=0.
            B_inf(Vm, hB, sB, pB, Bmin) =
                (1.0 - Bmin) / (1.0 + exp((Vm - hB) / sB))^pB + Bmin

            # ----------------------------
            # Appendix A4: voltage-dependent time constants
            # ----------------------------
            τ_1exp(Vm, tmax, tmin, ht, st) =
                (tmax - tmin) / (1.0 + exp((Vm - ht) / st)) + tmin

            τ_2exp(Vm, tmax, tmin, ht1, st1, ht2, st2) =
                (tmax - tmin) /
                ((1.0 + exp((Vm - ht1) / st1)) * (1.0 + exp((Vm - ht2) / st2))) + tmin

            # ----------------------------
            # Appendix A6/A8: Ca-dependent attenuation of Ca currents
            # ----------------------------
            gbr_inact(Ca_val, K) = Ca_val / (K + Ca_val)                  # Eq. A8
            f_att(Ca_val, K, b)  = 1.0 / (1.0 + b * gbr_inact(Ca_val, K)) # Eq. A6 (attenuation)

            # ----------------------------
            # Table 2: bilinear interpolation over the four corners
            # For binary PKA/PKC this selects exactly the appropriate column.
            # ----------------------------
            bilerp(x00, x10, x01, x11, pka, pkc) =
                (1 - pka) * (1 - pkc) * x00 +
                pka       * (1 - pkc) * x10 +
                (1 - pka) * pkc       * x01 +
                pka       * pkc       * x11
            # ===== Table 2: effective parameters under (PKA, PKC) =====
            gKS_V_eff  = bilerp(gKS_V_00,  gKS_V_10,  gKS_V_01,  gKS_V_11,  PKA, PKC)
            gKS_I_eff  = bilerp(gKS_I_00,  gKS_I_10,  gKS_I_01,  gKS_I_11,  PKA, PKC)
            gKV_eff    = bilerp(gKV_00,    gKV_10,    gKV_01,    gKV_11,    PKA, PKC)
            gCaL_eff   = bilerp(gCaL_00,   gCaL_10,   gCaL_01,   gCaL_11,   PKA, PKC)
            gKCaS_eff  = bilerp(gKCaS_00,  gKCaS_10,  gKCaS_01,  gKCaS_11,  PKA, PKC)

            tAmax_KV_eff = bilerp(tAmax_KV_00, tAmax_KV_10, tAmax_KV_01, tAmax_KV_11, PKA, PKC)
            tAmin_KV_eff = bilerp(tAmin_KV_00, tAmin_KV_10, tAmin_KV_01, tAmin_KV_11, PKA, PKC)
            tBmax_KV_eff = bilerp(tBmax_KV_00, tBmax_KV_10, tBmax_KV_01, tBmax_KV_11, PKA, PKC)
            tBmin_KV_eff = bilerp(tBmin_KV_00, tBmin_KV_10, tBmin_KV_01, tBmin_KV_11, PKA, PKC)

            I_bias_table = bilerp(I_bias_00, I_bias_10, I_bias_01, I_bias_11, PKA, PKC)

            # Total injected current (Appendix A1): network input + user injection + Table-2 bias + extra bias
            I_inj = jcn + I_stim + I_bias_table + I_bias_extra

            # ===== Conductances (Appendix A2 + A6/A8) =====
            g_Na   = gNa   * (A_Na^3)   * B_Na
            g_CaN  = gCaN  * (A_CaN^2)  * B_CaN * f_att(Ca, K_CaN_inact, b_Ca_inact)
            g_CaL  = gCaL_eff * (A_CaL^2) * B_CaL * f_att(Ca, K_CaL_inact, b_Ca_inact)
            g_KA_  = gKA   * (A_KA^3)   * B_KA
            g_KV_  = gKV_eff * (A_KV^2) * B_KV
            g_KS_V_ = gKS_V_eff * A_KS_V                      # activation-only
            g_KS_I_ = gKS_I_eff * A_inf(V, hA_KS_I, sA_KS_I)  # instantaneous
            g_KCaF_ = gKCaF * A_KCaF * max(gbr_F, 0.0)        # fast Ca-driven enhancement
            g_KCaS_ = gKCaS_eff * max(gbr_S, 0.0)             # slow Ca-driven enhancement
            g_L_    = gL

            # ===== Currents: I = g * (V - E) =====
            I_Na   = g_Na   * (V - E_Na)
            I_CaN  = g_CaN  * (V - E_Ca)
            I_CaL  = g_CaL  * (V - E_Ca)
            I_KA   = g_KA_  * (V - E_K)
            I_KV   = g_KV_  * (V - E_K)
            I_KS_V = g_KS_V_* (V - E_K)
            I_KS_I = g_KS_I_* (V - E_K)
            I_KCaF = g_KCaF_* (V - E_KCa)
            I_KCaS = g_KCaS_* (V - E_KCa)
            I_Lk   = g_L_   * (V - E_L)

            I_ion_sum = I_Na + I_CaN + I_CaL + I_KA + I_KV +
                        I_KS_V + I_KS_I + I_KCaF + I_KCaS + I_Lk
        end

        # --------------------------------------------------------
        # Appendix A1: membrane equation
        #   C_m dV/dt = I_inj - Σ I_ion
        # --------------------------------------------------------
        D(V) = (I_inj - I_ion_sum) / C_m

        # --------------------------------------------------------
        # Appendix A5: Ca pool
        #   dCa/dt = K_Ca * ( - (I_CaN + I_CaL) ) - Ca/τ_Ca
        # (Inward Ca currents are negative, so the minus makes Ca increase.)
        # --------------------------------------------------------
        D(Ca) = K_Ca * (-(I_CaN + I_CaL)) - Ca / τ_Ca

        # --------------------------------------------------------
        # Appendix A7: Ca-driven enhancer dynamics for KCa currents
        # --------------------------------------------------------
        D(gbr_F) = (Ca - gbr_F) / τ_gbr_F
        D(gbr_S) = (Ca - gbr_S) / τ_gbr_S

        # --------------------------------------------------------
        # Appendix A3/A4: gating dynamics
        #   dX/dt = (X_inf(V) - X) / τ_X(V)
        # --------------------------------------------------------
        D(A_Na) = (A_inf(V, hA_Na, sA_Na) - A_Na) /
                  τ_1exp(V, tAmax_Na, tAmin_Na, htA_Na, stA_Na)
        D(B_Na) = (B_inf(V, hB_Na, sB_Na, 1.0, 0.0) - B_Na) /
                  τ_1exp(V, tBmax_Na, tBmin_Na, htB_Na, stB_Na)

        D(A_CaN) = (A_inf(V, hA_CaN, sA_CaN) - A_CaN) /
                   τ_2exp(V, tAmax_CaN, tAmin_CaN,
                             htA1_CaN, stA1_CaN, htA2_CaN, stA2_CaN)
        D(B_CaN) = (B_inf(V, hB_CaN, sB_CaN, 1.0, 0.0) - B_CaN) /
                   τ_2exp(V, tBmax_CaN, tBmin_CaN,
                             htB1_CaN, stB1_CaN, htB2_CaN, stB2_CaN)

        D(A_CaL) = (A_inf(V, hA_CaL, sA_CaL) - A_CaL) /
                   τ_2exp(V, tAmax_CaL, tAmin_CaL,
                             htA1_CaL, stA1_CaL, htA2_CaL, stA2_CaL)
        D(B_CaL) = (B_inf(V, hB_CaL, sB_CaL, 1.0, 0.0) - B_CaL) /
                   τ_2exp(V, tBmax_CaL, tBmin_CaL,
                             htB1_CaL, stB1_CaL, htB2_CaL, stB2_CaL)

        D(A_KA) = (A_inf(V, hA_KA, sA_KA) - A_KA) /
                  τ_2exp(V, tAmax_KA, tAmin_KA,
                            htA1_KA, stA1_KA, htA2_KA, stA2_KA)
        D(B_KA) = (B_inf(V, hB_KA, sB_KA, 1.0, 0.0) - B_KA) /
                  τ_1exp(V, tBmax_KA, tBmin_KA, htB_KA, stB_KA)

        # IK-V: special inactivation with pB=2, Bmin=0.07 (Appendix note)
        D(A_KV) = (A_inf(V, hA_KV, sA_KV) - A_KV) /
                  τ_1exp(V, tAmax_KV_eff, tAmin_KV_eff, htA_KV, stA_KV)
        D(B_KV) = (B_inf(V, hB_KV, sB_KV, 2.0, 0.07) - B_KV) /
                  τ_2exp(V, tBmax_KV_eff, tBmin_KV_eff,
                            htB1_KV, stB1_KV, htB2_KV, stB2_KV)

        # IK,S-V: activation only; τA uses 2-exponential form
        D(A_KS_V) = (A_inf(V, hA_KS_V, sA_KS_V) - A_KS_V) /
                    τ_2exp(V, tAmax_KS_V, tAmin_KS_V,
                              htA1_KS_V, stA1_KS_V, htA2_KS_V, stA2_KS_V)

        # IK,Ca-F: activation only; τA constant
        D(A_KCaF) = (A_inf(V, hA_KCaF, sA_KCaF) - A_KCaF) / τA_KCaF
    end

    @computed_properties_with_inputs begin
        # Useful for debugging/plotting
        I_bias_table_effective = (1-PKA)*(1-PKC)*I_bias_00 +
                                 PKA*(1-PKC)*I_bias_10 +
                                 (1-PKA)*PKC*I_bias_01 +
                                 PKA*PKC*I_bias_11
        Ca_pool = Ca
    end
end

"""
    TRNNeuron(; C_M=1.0, g_Na=30.0, g_K=25.0, g_NaL=0.0247, g_KL=0.1855, g_ClL=0.49,
              g_TCa=1.75, g_KCa=10.0, g_CAN=0.25, g_NaCAN=0.124, g_KCAN=0.126, E_CAN=-20.0,
              I_NCXmax=1600.0, K_m_Na=87.5, K_m_Ca=1.38, k_sat=0.1, γ_NCX=0.35, F=96485.332, R=8.314, T=310.0,
              P_max=1.25, U_kcc2=0.3, G_glia=8.0, ε_k=0.33, K_bath=4.0, Na_gi=18.0,
              τ=0.001, γ_con=0.0445, RTF=26.64, RTF_Ca=13.32)

Thalamic reticular nucleus (TRN) neuron model with explicit ion concentration dynamics and coupling
to an α7 nAChR + ER Ca store module.
This neuron implements:
- a conductance-based membrane equation with Na/K/leak/Cl, T-type Ca, KCa, and CAN currents,
- explicit extracellular/intracellular ion concentration state variables (Na, K, Cl, Ca),
- an NCX exchanger current, Na/K pump, KCC2 cotransporter, glial uptake, and diffusion terms,
- Ca absorption, and
- external coupling inputs from an α7 receptor blox: `I_α7` (membrane current) and `J_ER` (Ca flux).

Membrane equation (sign conventions follow the implementation):
```math
\\frac{dV_m}{dt} = -\\frac{1}{C_M}\\left(I_{Na}+I_{K}+I_{ClL}+I_T+I_{NCX}+I_{KCa}+I_{CAN}+I_{\\alpha7}-I_{app}\\right)
```

Arguments:
- `C_M` [µF/cm²]: Membrane capacitance density.
- `g_*` [-]: Maximal conductances for the listed currents (units chosen so `I = g*(V-E)` yields µA/cm²).
- `E_CAN` [mV]: CAN reversal potential (fixed by the model table).
- NCX parameters: `I_NCXmax`, `K_m_Na`, `K_m_Ca`, `k_sat`, `γ_NCX`, `F`, `R`, `T`.
- Pump/transport parameters: `P_max`, `U_kcc2`, `G_glia`, `ε_k`, `K_bath`, `Na_gi`.
- Conversion factors: `τ` (ms→s), `γ_con` (current→concentration).
- Nernst factors: `RTF`, `RTF_Ca`.

Inputs:
- `I_app` [µA/cm²]: Applied current.
- `jcn` [µA/cm²]: Generic Neuroblox synaptic input (added to I_app).
- `I_α7` [µA/cm²]: α7 current from `Alpha7ERnAChR`.
- `J_ER` [mM/ms]: ER→cytosol Ca flux contribution from `Alpha7ERnAChR`.

Outputs:
- `V_m` [mV]: Membrane potential.
- `Ca_i` [mM]: Cytosolic calcium concentration (used by α7/ER module and Ca-dependent currents).

References:
1. King, J. R., et al. (2017). (α7 nAChR / Ca signaling framework used for the coupled receptor module). *Molecular Pharmacology*. (doi:10.1124/mol.117.111401)
"""
@blox struct TRNNeuron(;
    name,
    namespace = nothing,

    # ============================================================
    # Table 1 parameters (units follow your latex)
    # Conductances are in μA/(mV·cm^2) so that I = g*(V-E) is μA/cm^2
    # ============================================================
    C_M  = 1.0,        # μF/cm^2

    g_Na  = 30.0,
    g_K   = 25.0,
    g_NaL = 0.0247,
    g_KL  = 0.1855,    # (you can set 0.22062 for the adjusted case)
    g_ClL = 0.49,

    g_TCa = 1.75,
    g_KCa = 10.0,
    g_CAN = 0.25,
    g_NaCAN = 0.124,
    g_KCAN  = 0.126,

    # CAN reversal is fixed in your table
    E_CAN = -20.0,     # mV

    # NCX parameters
    I_NCXmax = 1600.0, # pA/pF (converted to μA/cm^2 by multiplying C_M; see below)
    K_m_Na   = 87.5,   # mM
    K_m_Ca   = 1.38,   # mM
    k_sat    = 0.1,
    γ_NCX    = 0.35,
    F = 96485.332,     # C/mol
    R = 8.314,         # J/(mol·K)
    T = 310.0,         # K (~37°C)

    # Pumps / transporters / diffusion / glia
    P_max  = 1.25,     # mM/s
    U_kcc2 = 0.3,      # mM/s
    G_glia = 8.0,      # mM/s
    ε_k    = 0.33,     # s^-1
    K_bath = 4.0,      # mM
    Na_gi  = 18.0,     # mM (glial Na, constant)

    # Conversion factors
    τ      = 0.001,    # ms->s
    γ_con  = 0.0445,   # current->concentration (μA/cm^2 -> mM/s)

    # Nernst factors (mV)
    RTF   = 26.64,     # monovalent
    RTF_Ca= 13.32      # divalent

) <: AbstractExciNeuron

    @params(
        C_M,
        g_Na, g_K, g_NaL, g_KL, g_ClL,
        g_TCa, g_KCa, g_CAN, g_NaCAN, g_KCAN, E_CAN,
        I_NCXmax, K_m_Na, K_m_Ca, k_sat, γ_NCX, F, R, T,
        P_max, U_kcc2, G_glia, ε_k, K_bath, Na_gi,
        τ, γ_con,
        RTF, RTF_Ca
    )

    @states(
        # Membrane
        V_m = -70.0,     # mV

        # Gating variables (your Table 2 provides some initials)
        m_H  = 0.0036,
        h_H  = 0.9992,
        j_H  = 0.0122,

        m_Ca  = 0.0,
        h_Ca  = 1.0,

        m_KCa = 0.0,
        m_CAN = 0.0,

        # Ionic concentrations (mM) — Table 2 initials
        Na_o = 144.0,
        Na_i = 18.0,
        K_o  = 4.0,
        K_i  = 140.0,
        Cl_o = 130.0,
        Cl_i = 6.0,
        Ca_i = 2.4e-4     # mM
    )

    @inputs(
        # Applied current in the membrane equation (μA/cm^2)
        I_app = 0.0,

        # Neuroblox generic synaptic input (μA/cm^2): we add it to I_app
        jcn   = 0.0,

        # From α7 receptor blox
        I_α7  = 0.0,      # μA/cm^2
        J_ER  = 0.0       # mM/ms (net ER->cyt contribution)
    )

    @outputs V_m Ca_i

    @equations begin
        @setup begin
            # ------------------------------------------------------------
            # Numerically-stable helper for rate expressions near 0/0
            # ------------------------------------------------------------
            vtrap_denom1(x, y) = abs(x / y) < 1e-7 ? (y * (1.0 + x/(2y))) : (x / (1.0 - exp(-x / y)))
            vtrap_expm1(x, y)  = abs(x / y) < 1e-7 ? (y * (1.0 - x/(2y))) : (x / (exp(x / y) - 1.0))

            # ------------------------------------------------------------
            # HH-like rates (from your equations)
            # ------------------------------------------------------------
            α_m(V) = 0.32 * vtrap_denom1(V + 54.0, 4.0)
            β_m(V) = 0.28 * vtrap_expm1(V + 27.0, 5.0)

            α_h(V) = 0.128 * exp(-(50.0 + V) / 18.0)
            β_h(V) = 4.0 / (1.0 + exp(-(V + 27.0) / 5.0))

            α_j(V) = 0.032 * vtrap_denom1(V + 52.0, 5.0)
            β_j(V) = 0.5 * exp(-(V + 57.0) / 40.0)

            # ------------------------------------------------------------
            # T-type Ca gating (your eqs 24–27)
            # ------------------------------------------------------------
            m_inf_Ca(V) = 1.0 / (1.0 + exp(-(V + 52.0) / 7.4))
            h_inf_Ca(V) = 1.0 / (1.0 + exp((V + 80.0) / 5.0))

            τ_mCa(V) = 0.44 + 0.15 / (exp((V + 27.0) / 10.0) + exp(-(V + 102.0) / 15.0))
            τ_hCa(V) = 22.7 + 0.27 / (exp((V + 48.0) / 4.0) + exp(-(V + 407.0) / 50.0))

            # ------------------------------------------------------------
            # Ca-dependent gating (your equations)
            # Ca_i is in mM here
            # ------------------------------------------------------------
            m_inf_KCa(Ca) = (48.0 * Ca^2) / (48.0 * Ca^2 + 0.03)
            τ_mKCa(Ca)    = 1.0 / (48.0 * Ca^2 + 0.03)

            m_inf_CAN(Ca) = (20.0 * Ca^2) / (20.0 * Ca^2 + 0.002)
            τ_mCAN(Ca)    = 1.0 / (20.0 * Ca^2 + 0.002)

            # ------------------------------------------------------------
            # Reversal potentials (mV) — your Nernst form
            # ------------------------------------------------------------
            E_Na = RTF    * log(max(Na_o, 1e-12) / max(Na_i, 1e-12))
            E_K  = RTF    * log(max(K_o,  1e-12) / max(K_i,  1e-12))
            E_Cl = RTF    * log(max(Cl_i, 1e-12) / max(Cl_o, 1e-12))
            E_Ca = RTF_Ca * log(2.0 / max(Ca_i, 1e-12))   # Ca_o fixed at 2.0 mM per your Table 2

            # ------------------------------------------------------------
            # Currents (μA/cm^2)
            # ------------------------------------------------------------
            I_Na   = g_Na  * (m_H^3) * h_H * (V_m - E_Na) + g_NaL * (V_m - E_Na)
            I_K    = g_K   * (j_H^4)       * (V_m - E_K)  + g_KL  * (V_m - E_K)
            I_ClL  = g_ClL * (V_m - E_Cl)

            I_T    = g_TCa * (m_Ca^2) * h_Ca * (V_m - E_Ca)

            I_KCa  = g_KCa * (m_KCa^2) * (V_m - E_K)

            # Membrane equation uses I_CAN with fixed E_CAN
            I_CAN  = g_CAN * (m_CAN^2) * (V_m - E_CAN)

            # Ion bookkeeping uses Na/K components of CAN
            I_NaCAN = g_NaCAN * (m_CAN^2) * (V_m - E_Na)
            I_KCAN  = g_KCAN  * (m_CAN^2) * (V_m - E_K)

            # ------------------------------------------------------------
            # NCX current (μA/cm^2)
            # Your table gives I_NCXmax in pA/pF.
            # Multiplying by C_M (μF/cm^2) converts pA/pF -> μA/cm^2 (unit cancellation).
            # ------------------------------------------------------------
            V_volt = V_m * 1e-3
            exp1 = exp((γ_NCX)     * V_volt * F / (R * T))
            exp2 = exp((γ_NCX - 1) * V_volt * F / (R * T))

            num = exp1 * (Na_i^3) * 2.0 - exp2 * (Na_o^3) * Ca_i   # Ca_o=2.0 mM
            den = (K_m_Na^3 + Na_o^3) * (K_m_Ca + 2.0) * (1.0 + k_sat * exp2)

            I_NCX = (C_M * I_NCXmax) * (num / max(den, 1e-18))

            # ------------------------------------------------------------
            # Pumps / transporters / glia (mM/s)
            # ------------------------------------------------------------
            I_pump = (P_max / (1.0 + exp((25.0 - Na_i) / 3.0))) *
                     (1.0  / (1.0 + exp(5.5 - K_o)))

            I_gliapump = (1.0/3.0) *
                         (P_max / (1.0 + exp((25.0 - Na_gi) / 3.0))) *
                         (1.0  / (1.0 + exp(5.5 - K_o)))

            I_glia = G_glia / (1.0 + exp((18.0 - K_o) / 2.5))
            I_diff = ε_k * (K_o - K_bath)

            I_kcc2 = U_kcc2 * log( (max(K_i,1e-12) * max(Cl_i,1e-12)) /
                                   (max(K_o,1e-12) * max(Cl_o,1e-12)) )

            # ------------------------------------------------------------
            # Membrane potential ODE (STRICT sign as your equation)
            # ------------------------------------------------------------
            I_app_total = I_app + jcn

            # Ca absorption term: Ca_absp = Ca_i / 5  (mM/s)
            Ca_absp = Ca_i / 5.0
        end

        D(V_m) = -(1.0 / C_M) * (I_Na + I_K + I_ClL + I_T + I_NCX + I_KCa + I_CAN + I_α7 - I_app_total)

        # ------------------------------------------------------------
        # Gating ODEs
        # ------------------------------------------------------------
        D(m_H) = α_m(V_m) * (1.0 - m_H) - β_m(V_m) * m_H
        D(h_H) = α_h(V_m) * (1.0 - h_H) - β_h(V_m) * h_H
        D(j_H) = α_j(V_m) * (1.0 - j_H) - β_j(V_m) * j_H

        D(m_Ca) = (m_inf_Ca(V_m) - m_Ca) / τ_mCa(V_m)
        D(h_Ca) = (h_inf_Ca(V_m) - h_Ca) / τ_hCa(V_m)

        D(m_KCa) = (m_inf_KCa(Ca_i) - m_KCa) / τ_mKCa(Ca_i)
        D(m_CAN) = (m_inf_CAN(Ca_i) - m_CAN) / τ_mCAN(Ca_i)

        # ------------------------------------------------------------
        # Ion dynamics (mM/ms) — STRICTLY your dN/dt forms collapsed to concentrations
        # ------------------------------------------------------------
        D(Na_o) = τ * ( γ_con * I_Na + 3.0 * γ_con * I_NCX + γ_con * I_NaCAN + 3.0 * I_pump )
        D(Na_i) = τ * ( -γ_con * I_Na - 3.0 * γ_con * I_NCX - γ_con * I_NaCAN - 3.0 * I_pump )

        D(K_o)  = τ * ( γ_con * I_K + γ_con * I_KCa + γ_con * I_KCAN
                        - 2.0 * I_pump + I_kcc2 - I_diff - I_glia - 2.0 * I_gliapump )

        D(K_i)  = τ * ( -γ_con * I_K - γ_con * I_KCa - γ_con * I_KCAN
                        + 2.0 * I_pump - I_kcc2 )

        D(Cl_o) = τ * ( -γ_con * I_ClL + I_kcc2 )
        D(Cl_i) = τ * (  γ_con * I_ClL - I_kcc2 )

        #   Ca_i is owned by neuron. It receives:
        #   - membrane fluxes: I_T and I_α7 (both converted by γ_con)
        #   - NCX term: +γ_con * I_NCX
        #   - absorption: -Ca_absp
        #   - ER contribution: +J_ER (already mM/ms)
        D(Ca_i) = τ * ( -γ_con * (I_T + I_α7) + γ_con * I_NCX - Ca_absp ) + J_ER
    end
end





"""
    MuscarinicNeuron(; C_M=1.0, g_Na=30.0, g_K=25.0, g_NaL=0.0247, g_KL=0.1855, g_ClL=0.49,
                     g_TCa=1.75, g_KCa=10.0, g_CAN=0.25, g_NaCAN=0.124, g_KCAN=0.126, E_CAN=-20.0,
                     I_NCXmax=1600.0, K_m_Na=87.5, K_m_Ca=1.38, k_sat=0.1, γ_NCX=0.35, F=96485.332, R=8.314, T=310.0,
                     P_max=1.25, U_kcc2=0.3, G_glia=8.0, ε_k=0.33, K_bath=4.0, Na_gi=18.0,
                     τ=0.001, γ_con=0.0445, RTF=26.64, RTF_Ca=13.32)

Thalamic reticular nucleus (TRN)-style conductance-based neuron with explicit ion concentration
dynamics, extended with a muscarinic-activated NCM/INCM current input that supports Na/K
apportionment for ion conservation.

This neuron is a *new* neuron type (keeps your original `TRNNeuron` unchanged) and adds three
input ports intended to be wired from `MuscarinicR`:
- `I_NCM`   : total muscarinic NCM current (added to the membrane current balance),
- `I_NaNCM` : Na portion of that current (added to Na ion bookkeeping),
- `I_KNCM`  : K portion of that current (added to K ion bookkeeping).

Key design choices:
- NCM is treated as Ca-impermeable, so it does not appear in the Ca equation.
- If `I_NCM` is large and you are conserving ions explicitly, using `I_NaNCM`/`I_KNCM` avoids
  violating Na/K mass balance by accounting for the ionic carriers of the NCM current.

Membrane equation (sign conventions follow the implementation):
```math
\\frac{dV_m}{dt} = -\\frac{1}{C_M}\\left(I_{Na}+I_{K}+I_{ClL}+I_T+I_{NCX}+I_{KCa}+I_{CAN}+I_{\\alpha7}+I_{NCM}-I_{app}\\right)
```

Arguments:
- `C_M` [µF/cm²]: Membrane capacitance density.
- `g_*` [-]: Maximal conductances for Na/K/leak/Cl, T-type Ca, KCa, and CAN currents (units chosen so `I = g*(V-E)` yields µA/cm²).
- `E_CAN` [mV]: CAN reversal potential (fixed by the model table).
- NCX parameters: `I_NCXmax`, `K_m_Na`, `K_m_Ca`, `k_sat`, `γ_NCX`, `F`, `R`, `T`.
- Pump/transport parameters: `P_max`, `U_kcc2`, `G_glia`, `ε_k`, `K_bath`, `Na_gi`.
- Conversion factors: `τ` (ms→s), `γ_con` (µA/cm² → mM/s).
- Nernst factors: `RTF` (monovalent), `RTF_Ca` (divalent).

Inputs:
- `I_app` [µA/cm²]: Applied current.
- `jcn` [µA/cm²]: Generic Neuroblox synaptic input (added to `I_app`).
- `I_α7` [µA/cm²]: α7 current from an α7 receptor module (if used).
- `J_ER` [mM/ms]: ER→cytosol Ca flux contribution from an ER Ca store module (if used).
- `I_NCM` [µA/cm²]: Total muscarinic NCM/INCM current from `MuscarinicR`.
- `I_NaNCM` [µA/cm²]: Na portion of I_NCM for Na ion bookkeeping.
- `I_KNCM` [µA/cm²]: K portion of I_NCM for K ion bookkeeping.

Outputs:
- `V_m` [mV]: Membrane potential.
- `Ca_i` [mM]: Cytosolic calcium concentration.

References:
1. Fransen, E., Alonso, A. A., & Hasselmo, M. E. (2002). Simulations of the Role of the Muscarinic-Activated Calcium-Sensitive Nonspecific Cation Current INCM in Entorhinal Neuronal Activity during Delayed Matching Tasks. *Journal of Neuroscience*, 22(3), 1081-1097.
"""
@blox struct MuscarinicNeuron(;
    name,
    namespace = nothing,

    # ============================================================
    # Table 1 parameters
    # Conductances are in μA/(mV·cm^2) so that I = g*(V-E) is μA/cm^2
    # ============================================================
    C_M  = 1.0,        # μF/cm^2

    g_Na  = 30.0,
    g_K   = 25.0,
    g_NaL = 0.0247,
    g_KL  = 0.1855,
    g_ClL = 0.49,

    g_TCa = 1.75,
    g_KCa = 10.0,
    g_CAN = 0.25,
    g_NaCAN = 0.124,
    g_KCAN  = 0.126,

    E_CAN = -20.0,     # mV

    # NCX parameters
    I_NCXmax = 1600.0, # pA/pF
    K_m_Na   = 87.5,   # mM
    K_m_Ca   = 1.38,   # mM
    k_sat    = 0.1,
    γ_NCX    = 0.35,
    F = 96485.332,     # C/mol
    R = 8.314,         # J/(mol·K)
    T = 310.0,         # K

    # Pumps / transporters / diffusion / glia
    P_max  = 1.25,     # mM/s
    U_kcc2 = 0.3,      # mM/s
    G_glia = 8.0,      # mM/s
    ε_k    = 0.33,     # s^-1
    K_bath = 4.0,      # mM
    Na_gi  = 18.0,     # mM

    # Conversion factors
    τ      = 0.001,    # ms->s
    γ_con  = 0.0445,   # (μA/cm^2 -> mM/s)

    # Nernst factors (mV)
    RTF   = 26.64,
    RTF_Ca= 13.32
) <: AbstractExciNeuron

    @params(
        C_M,
        g_Na, g_K, g_NaL, g_KL, g_ClL,
        g_TCa, g_KCa, g_CAN, g_NaCAN, g_KCAN, E_CAN,
        I_NCXmax, K_m_Na, K_m_Ca, k_sat, γ_NCX, F, R, T,
        P_max, U_kcc2, G_glia, ε_k, K_bath, Na_gi,
        τ, γ_con,
        RTF, RTF_Ca
    )

    @states(
        # Membrane
        V_m = -70.0,     # mV

        # Gating variables
        m_H  = 0.0036,
        h_H  = 0.9992,
        j_H  = 0.0122,

        m_Ca  = 0.0,
        h_Ca  = 1.0,

        m_KCa = 0.0,
        m_CAN = 0.0,

        # Ionic concentrations (mM)
        Na_o = 144.0,
        Na_i = 18.0,
        K_o  = 4.0,
        K_i  = 140.0,
        Cl_o = 130.0,
        Cl_i = 6.0,
        Ca_i = 2.4e-4
    )

    @inputs(
        # Applied current
        I_app = 0.0,

        # Neuroblox generic synaptic input
        jcn   = 0.0,

        # From α7 receptor blox
        I_α7  = 0.0,      # µA/cm^2
        J_ER  = 0.0,      # mM/ms

        # ------------------------------------------------------------
        # NEW: muscarinic INCM/NCM inputs with Na/K apportionment
        # ------------------------------------------------------------
        I_NCM   = 0.0,    # µA/cm^2 (total)
        I_NaNCM = 0.0,    # µA/cm^2 (Na portion)
        I_KNCM  = 0.0     # µA/cm^2 (K portion)
    )

    @outputs V_m Ca_i

    @equations begin
        @setup begin
            # Numerically-stable helpers
            vtrap_denom1(x, y) = abs(x / y) < 1e-7 ? (y * (1.0 + x/(2y))) : (x / (1.0 - exp(-x / y)))
            vtrap_expm1(x, y)  = abs(x / y) < 1e-7 ? (y * (1.0 - x/(2y))) : (x / (exp(x / y) - 1.0))

            # HH-like rates
            α_m(V) = 0.32 * vtrap_denom1(V + 54.0, 4.0)
            β_m(V) = 0.28 * vtrap_expm1(V + 27.0, 5.0)

            α_h(V) = 0.128 * exp(-(50.0 + V) / 18.0)
            β_h(V) = 4.0 / (1.0 + exp(-(V + 27.0) / 5.0))

            α_j(V) = 0.032 * vtrap_denom1(V + 52.0, 5.0)
            β_j(V) = 0.5 * exp(-(V + 57.0) / 40.0)

            # T-type Ca gating
            m_inf_Ca(V) = 1.0 / (1.0 + exp(-(V + 52.0) / 7.4))
            h_inf_Ca(V) = 1.0 / (1.0 + exp((V + 80.0) / 5.0))

            τ_mCa(V) = 0.44 + 0.15 / (exp((V + 27.0) / 10.0) + exp(-(V + 102.0) / 15.0))
            τ_hCa(V) = 22.7 + 0.27 / (exp((V + 48.0) / 4.0) + exp(-(V + 407.0) / 50.0))

            # Ca-dependent gating
            m_inf_KCa(Ca) = (48.0 * Ca^2) / (48.0 * Ca^2 + 0.03)
            τ_mKCa(Ca)    = 1.0 / (48.0 * Ca^2 + 0.03)

            m_inf_CAN(Ca) = (20.0 * Ca^2) / (20.0 * Ca^2 + 0.002)
            τ_mCAN(Ca)    = 1.0 / (20.0 * Ca^2 + 0.002)

            # Reversal potentials (mV)
            E_Na = RTF    * log(max(Na_o, 1e-12) / max(Na_i, 1e-12))
            E_K  = RTF    * log(max(K_o,  1e-12) / max(K_i,  1e-12))
            E_Cl = RTF    * log(max(Cl_i, 1e-12) / max(Cl_o, 1e-12))
            E_Ca = RTF_Ca * log(2.0 / max(Ca_i, 1e-12))   # Ca_o fixed at 2.0 mM

            # Currents (μA/cm^2)
            I_Na   = g_Na  * (m_H^3) * h_H * (V_m - E_Na) + g_NaL * (V_m - E_Na)
            I_K    = g_K   * (j_H^4)       * (V_m - E_K)  + g_KL  * (V_m - E_K)
            I_ClL  = g_ClL * (V_m - E_Cl)

            I_T    = g_TCa * (m_Ca^2) * h_Ca * (V_m - E_Ca)
            I_KCa  = g_KCa * (m_KCa^2) * (V_m - E_K)

            # CAN (membrane uses fixed E_CAN)
            I_CAN  = g_CAN * (m_CAN^2) * (V_m - E_CAN)

            # Na/K components of CAN for bookkeeping
            I_NaCAN = g_NaCAN * (m_CAN^2) * (V_m - E_Na)
            I_KCAN  = g_KCAN  * (m_CAN^2) * (V_m - E_K)

            # NCX current
            V_volt = V_m * 1e-3
            exp1 = exp((γ_NCX)     * V_volt * F / (R * T))
            exp2 = exp((γ_NCX - 1) * V_volt * F / (R * T))

            num = exp1 * (Na_i^3) * 2.0 - exp2 * (Na_o^3) * Ca_i
            den = (K_m_Na^3 + Na_o^3) * (K_m_Ca + 2.0) * (1.0 + k_sat * exp2)

            I_NCX = (C_M * I_NCXmax) * (num / max(den, 1e-18))

            # Pumps / transporters / glia (mM/s)
            I_pump = (P_max / (1.0 + exp((25.0 - Na_i) / 3.0))) *
                     (1.0  / (1.0 + exp(5.5 - K_o)))

            I_gliapump = (1.0/3.0) *
                         (P_max / (1.0 + exp((25.0 - Na_gi) / 3.0))) *
                         (1.0  / (1.0 + exp(5.5 - K_o)))

            I_glia = G_glia / (1.0 + exp((18.0 - K_o) / 2.5))
            I_diff = ε_k * (K_o - K_bath)

            I_kcc2 = U_kcc2 * log( (max(K_i,1e-12) * max(Cl_i,1e-12)) /
                                   (max(K_o,1e-12) * max(Cl_o,1e-12)) )

            # Applied + synaptic
            I_app_total = I_app + jcn

            # Ca absorption (mM/s)
            Ca_absp = Ca_i / 5.0
        end

        # ============================================================
        # Membrane potential ODE
        # NEW: add I_NCM to the current balance
        # ============================================================
        D(V_m) = -(1.0 / C_M) * (I_Na + I_K + I_ClL + I_T + I_NCX + I_KCa + I_CAN + I_α7 + I_NCM - I_app_total)

        # ============================================================
        # Gating ODEs
        # ============================================================
        D(m_H) = α_m(V_m) * (1.0 - m_H) - β_m(V_m) * m_H
        D(h_H) = α_h(V_m) * (1.0 - h_H) - β_h(V_m) * h_H
        D(j_H) = α_j(V_m) * (1.0 - j_H) - β_j(V_m) * j_H

        D(m_Ca) = (m_inf_Ca(V_m) - m_Ca) / τ_mCa(V_m)
        D(h_Ca) = (h_inf_Ca(V_m) - h_Ca) / τ_hCa(V_m)

        D(m_KCa) = (m_inf_KCa(Ca_i) - m_KCa) / τ_mKCa(Ca_i)
        D(m_CAN) = (m_inf_CAN(Ca_i) - m_CAN) / τ_mCAN(Ca_i)

        # ============================================================
        # Ion dynamics (mM/ms)
        # NEW: Na/K apportionment terms I_NaNCM, I_KNCM
        # ============================================================
        D(Na_o) = τ * ( γ_con * I_Na + 3.0 * γ_con * I_NCX + γ_con * I_NaCAN + γ_con * I_NaNCM + 3.0 * I_pump )
        D(Na_i) = τ * ( -γ_con * I_Na - 3.0 * γ_con * I_NCX - γ_con * I_NaCAN - γ_con * I_NaNCM - 3.0 * I_pump )

        D(K_o)  = τ * ( γ_con * I_K + γ_con * I_KCa + γ_con * I_KCAN + γ_con * I_KNCM
                        - 2.0 * I_pump + I_kcc2 - I_diff - I_glia - 2.0 * I_gliapump )

        D(K_i)  = τ * ( -γ_con * I_K - γ_con * I_KCa - γ_con * I_KCAN - γ_con * I_KNCM
                        + 2.0 * I_pump - I_kcc2 )

        D(Cl_o) = τ * ( -γ_con * I_ClL + I_kcc2 )
        D(Cl_i) = τ * (  γ_con * I_ClL - I_kcc2 )

        # Ca_i (owned by neuron): unchanged by INCM/NCM (Ca-impermeable)
        D(Ca_i) = τ * ( -γ_con * (I_T + I_α7) + γ_con * I_NCX - Ca_absp ) + J_ER
    end
end


"""
    VTADANeuron(; C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, I_bias=0.0)

VTA dopaminergic (DA) neuron model for use with β2-nAChR receptors.

Implements a Hodgkin-Huxley-type neuron with sodium, potassium, and leak currents,
plus an input for nAChR-mediated current. Uses unified outward-positive current convention.

```math
C_m \\frac{dV}{dt} = I_{app} - I_{Na} - I_K - I_L - I_{ACh}
```

Arguments:
- `C_m` [µF/cm²]: Membrane capacitance.
- `g_Na`, `g_K`, `g_L` [mS/cm²]: Maximal conductances.
- `E_Na`, `E_K`, `E_L` [mV]: Reversal potentials.
- `I_bias` [µA/cm²]: Constant bias current.

Inputs:
- `I_app` [µA/cm²]: Applied current.
- `I_ACh` [µA/cm²]: nAChR current (outward positive).

Outputs:
- `V` [mV]: Membrane potential.

References:
1. Morozova, E. O., et al. (2020). Distinct Temporal Structure of Nicotinic ACh Receptor Activation Determines Responses of VTA Neurons to Endogenous ACh and Nicotine. *eNeuro*, 7(4). (doi:10.1523/ENEURO.0418-19.2020)
"""
@blox struct VTADANeuron(;
    name,
    namespace=nothing,
    C_m=1.0,
    g_Na=120.0,
    g_K=36.0,
    g_L=0.3,
    E_Na=50.0,
    E_K=-77.0,
    E_L=-54.4,
    I_bias=0.0
) <: AbstractExciNeuron

    @params(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_bias)

    @states(
        V = -65.0,
        m = 0.05,
        h = 0.6,
        n = 0.32
    )

    @inputs(
        I_app = 0.0,
        I_ACh = 0.0
    )

    @outputs V

    @equations begin
        @setup begin
            α_m(V) = ifelse(abs(V + 40.0) < 1e-7, 1.0, 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0)))
            β_m(V) = 4.0 * exp(-(V + 65.0) / 18.0)
            α_h(V) = 0.07 * exp(-(V + 65.0) / 20.0)
            β_h(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
            α_n(V) = ifelse(abs(V + 55.0) < 1e-7, 0.1, 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0)))
            β_n(V) = 0.125 * exp(-(V + 65.0) / 80.0)

            I_Na = g_Na * (m^3) * h * (V - E_Na)
            I_K = g_K * (n^4) * (V - E_K)
            I_L = g_L * (V - E_L)
            I_total = I_app + I_bias
        end

        D(V) = (I_total - I_Na - I_K - I_L - I_ACh) / C_m
        D(m) = α_m(V) * (1.0 - m) - β_m(V) * m
        D(h) = α_h(V) * (1.0 - h) - β_h(V) * h
        D(n) = α_n(V) * (1.0 - n) - β_n(V) * n
    end
end


"""
    VTAGABANeuron(; C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, I_bias=0.0)

VTA GABAergic neuron model for use with β2-nAChR receptors.

Identical to VTADANeuron but typed as AbstractInhNeuron for network wiring.

Arguments:
- `C_m` [µF/cm²]: Membrane capacitance.
- `g_Na`, `g_K`, `g_L` [mS/cm²]: Maximal conductances.
- `E_Na`, `E_K`, `E_L` [mV]: Reversal potentials.
- `I_bias` [µA/cm²]: Constant bias current.

Inputs:
- `I_app` [µA/cm²]: Applied current.
- `I_ACh` [µA/cm²]: nAChR current (outward positive).

Outputs:
- `V` [mV]: Membrane potential.

References:
1. Morozova, E. O., et al. (2020). Distinct Temporal Structure of Nicotinic ACh Receptor Activation Determines Responses of VTA Neurons to Endogenous ACh and Nicotine. *eNeuro*, 7(4). (doi:10.1523/ENEURO.0418-19.2020)
"""
@blox struct VTAGABANeuron(;
    name,
    namespace=nothing,
    C_m=1.0,
    g_Na=120.0,
    g_K=36.0,
    g_L=0.3,
    E_Na=50.0,
    E_K=-77.0,
    E_L=-54.4,
    I_bias=0.0
) <: AbstractInhNeuron

    @params(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_bias)

    @states(
        V = -65.0,
        m = 0.05,
        h = 0.6,
        n = 0.32
    )

    @inputs(
        I_app = 0.0,
        I_ACh = 0.0
    )

    @outputs V

    @equations begin
        @setup begin
            α_m(V) = ifelse(abs(V + 40.0) < 1e-7, 1.0, 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0)))
            β_m(V) = 4.0 * exp(-(V + 65.0) / 18.0)
            α_h(V) = 0.07 * exp(-(V + 65.0) / 20.0)
            β_h(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
            α_n(V) = ifelse(abs(V + 55.0) < 1e-7, 0.1, 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0)))
            β_n(V) = 0.125 * exp(-(V + 65.0) / 80.0)

            I_Na = g_Na * (m^3) * h * (V - E_Na)
            I_K = g_K * (n^4) * (V - E_K)
            I_L = g_L * (V - E_L)
            I_total = I_app + I_bias
        end

        D(V) = (I_total - I_Na - I_K - I_L - I_ACh) / C_m
        D(m) = α_m(V) * (1.0 - m) - β_m(V) * m
        D(h) = α_h(V) * (1.0 - h) - β_h(V) * h
        D(n) = α_n(V) * (1.0 - n) - β_n(V) * n
    end
end
