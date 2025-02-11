abstract type AbstractInhNeuronBlox <: AbstractNeuronBlox end
abstract type AbstractExciNeuronBlox <: AbstractNeuronBlox end

struct HHNeuronExciBlox <: AbstractExciNeuronBlox
    system
    namespace

	function HHNeuronExciBlox(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        G_syn=3, 
        I_bg=0.0,
        freq=0,
        phase=0,
        τ=5
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
			G(t)=0.0 
            [output=true]
			z(t)=0.0
			Gₛₜₚ(t)=0.0 
            spikes_cumulative(t)=0.0
            spikes_window(t)=0.0
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 52 
			G_K  = 20 
			G_L = 0.1 
			E_Na = 55 
			E_K = -90 
			E_L = -60 
			G_syn = G_syn 
			V_shift = 10 
			V_range = 35 
			τ₁ = 0.1 
			τ₂ = τ 
			τ₃ = 2000 
			I_bg=I_bg
			kₛₜₚ = 0.5
			freq = freq 
			phase = phase
            spikes = 0
			spk_const = 1.127
		end

		αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
	    βₙ(v) = 0.125*exp(-(v+44)/80)
	    αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
	    βₘ(v) = 4*exp(-(v+55)/18)
		αₕ(v) = 0.07*exp(-(v+44)/20)
	    βₕ(v) = 1/(1+exp(-(v+14)/10))
		ϕ = 5 
		G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
		eqs = [ 
			   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in, 
			   D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(G)~(-1/τ₂)*G + z,
			   D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
			   D(Gₛₜₚ)~(-1/τ₃)*Gₛₜₚ + (z/5)*(kₛₜₚ-Gₛₜₚ),
               # HACK : need to define a Differential equation for spike counting
               # the alternative of having it as an algebraic equation with [irreducible=true]
               # leads to incorrect or unstable solutions. Needs more attention!
               D(spikes_cumulative) ~ spk_const*G_asymp(V,G_syn),
               D(spikes_window) ~ spk_const*G_asymp(V,G_syn)
		]
        
		sys = ODESystem(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	

struct HHNeuronInhibBlox <: AbstractInhNeuronBlox
    system
    namespace
	function HHNeuronInhibBlox(;
        name, 
        namespace = nothing, 
        E_syn=-70.0,
        G_syn=11.5,
        I_bg=0.0,
        freq=0,
        phase=0,
        τ=70
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)
			[input=true] 
			I_asc(t)
			[input=true]
			I_in(t)
			[input=true]
            G(t)=0.0 
			[output=true] 
			z(t)=0.0
            spikes_cumulative(t)=0.0
            spikes_window(t)=0.0
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 52 
			G_K  = 20 
			G_L = 0.1 
			E_Na = 55 
			E_K = -90 
			E_L = -60 
			G_syn = G_syn 
			V_shift = 0 
			V_range = 35 
			τ₁ = 0.1 
			τ₂ = τ 
			τ₃ = 2000 
			I_bg=I_bg 
			freq = freq 
			phase = phase
			spk_const = 1.127
		end

	   	αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
		βₙ(v) = 0.125*exp(-(v+48)/80)
        αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
		βₘ(v) = 4*exp(-(v+58)/18)
        αₕ(v) = 0.07*exp(-(v+51)/20)
		βₕ(v) = 1/(1+exp(-(v+21)/10))   	
		ϕ = 5 
		G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
	 	eqs = [ 
			   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in, 
			   D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(G)~(-1/τ₂)*G + z,
			   D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
		]

        sys = ODESystem(
            eqs, t, sts, ps; 
			name = Symbol(name)
        )
        
		new(sys, namespace)
	end
end	

#These neurons were used in Adam et. al 2022 model for DBS

struct HHNeuronInhib_MSN_Adam_Blox <: AbstractInhNeuronBlox
    system
    namespace

	function HHNeuronInhib_MSN_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=1.172,
        freq=0,
        phase=0,
        τ=13,
        Cₘ=1.0,
		σ=0.11,
		a=2,
		b=4,
		T=37,
		G_M=1.3
    )
		sts = @variables begin 
			V(t)=-63.83 
			n(t)=0.062
			m(t)=0.027
			h(t)=0.99
			mM(t)=0.022
			I_syn(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
			G(t)=0.0 
			[output =true] 
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 100 
			G_K  = 80 
			G_L = 0.1 
			G_M = G_M
			E_Na = 50 
			E_K = -100 
			E_L = -67 
			I_bg=I_bg
			freq = freq 
			phase = phase
           	Cₘ = Cₘ
			σ = σ
			a = a
			b = b
			T = T
		end
        
        @brownian χ

		αₙ(v) = 0.032*(v+52)/(1-exp(-(v+52)/5))
	    βₙ(v) = 0.5*exp(-(v+57)/40)
	    αₘ(v) = 0.32*(v+54)/(1-exp(-(v+54)/4))
	    βₘ(v) = 0.28*(v+27)/(exp((v+27)/5)-1)
		αₕ(v) = 0.128*exp(-(v+50)/18)
	    βₕ(v) = 4/(1+exp(-(v+27)/5))
		αₘₘ(v) = Q(T)*10^(-4)*(v+30)/(1-exp(-(v+30)/9))
		βₘₘ(v) = -Q(T)*10^(-4)*(v+30)/(1-exp((v+30)/9))

		Q(T) = 2.3^((T-23)/10)
		
		G_asymp(v,a,b) = a*(1+tanh(v/b))
		
		eqs = [ 
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)-G_M*mM*(V-E_K)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in+σ*χ), 
			   D(n)~(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(mM)~(αₘₘ(V)*(1-mM)-βₘₘ(V)*mM),
			   D(G)~(-1/τ)*G + G_asymp(V,a,b)*(1-G)
			  
		]
        
		sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	

struct HHNeuronInhib_FSI_Adam_Blox <: AbstractInhNeuronBlox
    system
    namespace

	function HHNeuronInhib_FSI_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=6.2,
        freq=0,
        phase=0,
        τ=11,
		τₛ=6.5,
        Cₘ=1.0,
		σ=1.2,
		a=4,
		b=10,
		T=37
    )
		sts = @variables begin 
			V(t)=-70.00 
			n(t)=0.032 
			h(t)=0.059 
			mD(t)=0.05
			hD(t)=0.059
			I_syn(t)
			[input=true] 
			I_gap(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
			G(t)=0.0 
			[output=true] 
			Gₛ(t)=0.0 
			[output=true] 
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 112.5 
			G_K  = 225 
			G_L = 0.25 
			G_D = 6
			E_Na = 50 
			E_K = -90 
			E_L = -70 
			I_bg=I_bg
			freq = freq 
			phase = phase
           	Cₘ = Cₘ
			σ = σ
			a = a
			b = b
			T = T
            τ = τ
            τₛ = τₛ
		end
        
        @brownian χ

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

		eqs = [ 
			   D(V)~(1/Cₘ)*(-G_Na*m_inf(V)^3*h*(V-E_Na)-G_K*n^2*(V-E_K)-G_L*(V-E_L)-G_D*mD^3*hD*(V-E_K)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_gap+I_asc+I_in+σ*χ), 
			   D(n)~(n_inf(V)-n)/τₙ(V), 
			   D(h)~(h_inf(V)-h)/τₕ(V),
			   D(mD)~(mD_inf(V)-mD)/τₘD(V),
			   D(hD)~(hD_inf(V)-hD)/τₕD(V),
			   D(G)~(-1/τ)*G + G_asymp(V,a,b)*(1-G),
			   D(Gₛ)~(-1/τₛ)*Gₛ + G_asymp(V,a,b)*(1-Gₛ)
		]
        
		sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	

struct HHNeuronExci_STN_Adam_Blox <: AbstractExciNeuronBlox
    system
    namespace

	function HHNeuronExci_STN_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        I_bg=1.8,
        freq=0,
        phase=0,
        τ=2,
        Cₘ=1.0,
		σ=1.7,
		a=5,
		b=4
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
			DBS_in(t)
			[input=true]
			G(t)=0.0 
			[output = true] 
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 100 
			G_K  = 80 
			G_L = 0.1 
			E_Na = 50 
			E_K = -100 
			E_L = -67 
			I_bg=I_bg
			freq = freq 
			phase = phase
           	Cₘ = Cₘ
			σ = σ
			a = a
			b = b
		end
        
        @brownian χ

		αₙ(v) = 0.032*(v+52)/(1-exp(-(v+52)/5))
	    βₙ(v) = 0.5*exp(-(v+57)/40)
	    αₘ(v) = 0.32*(v+54)/(1-exp(-(v+54)/4))
	    βₘ(v) = 0.28*(v+27)/(exp((v+27)/5)-1)
		αₕ(v) = 0.128*exp(-(v+50)/18)
	    βₕ(v) = 4/(1+exp(-(v+27)/5))
		
		G_asymp(v,a,b) = a*(1+tanh(v/b + DBS_in))
		
		eqs = [ 
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in+σ*χ), 
			   D(n)~(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(G)~(-1/τ)*G + G_asymp(V,a,b)*(1-G)
			  
		]
        
		sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	

struct HHNeuronInhib_GPe_Adam_Blox <: AbstractInhNeuronBlox
    system
    namespace

	function HHNeuronInhib_GPe_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=3.4,
        freq=0,
        phase=0,
        τ=10,
        Cₘ=1.0,
		σ=1.7,
		a=2,
		b=4,
		T=37
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
			G(t)=0.0 
			[output = true] 
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 100 
			G_K  = 80 
			G_L = 0.1 
			G_M = 1.3
			E_Na = 50 
			E_K = -100 
			E_L = -67 
			I_bg=I_bg
			freq = freq 
			phase = phase
           	Cₘ = Cₘ
			σ = σ
			a = a
			b = b
			T = T
		end
        
        @brownian χ

		αₙ(v) = 0.032*(v+52)/(1-exp(-(v+52)/5))
	    βₙ(v) = 0.5*exp(-(v+57)/40)
	    αₘ(v) = 0.32*(v+54)/(1-exp(-(v+54)/4))
	    βₘ(v) = 0.28*(v+27)/(exp((v+27)/5)-1)
		αₕ(v) = 0.128*exp(-(v+50)/18)
	    βₕ(v) = 4/(1+exp(-(v+27)/5))
		
		
		G_asymp(v,a,b) = a*(1+tanh(v/b))
		
		eqs = [ 
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in+σ*χ), 
			   D(n)~(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(G)~(-1/τ)*G + G_asymp(V,a,b)*(1-G)			  
		]
        
		sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	



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
- C: Membrane capicitance (μF).
- θ: Threshold voltage (mV).
- Eₘ: Resting membrane potential (mV).
- I_in: External current input (μA).

References:
1. Abbott, L. Lapicque's introduction of the integrate-and-fire model neuron (1907). Brain Res Bull 50, 303-304 (1999).
"""

# Paramater bounds for GUI
# C = [0.1, 100] μF
# θ = [-65, -45] mV
# Eₘ = [-100, -55] mV - If Eₘ >= θ obvious instability
# I_in = [-2.5, 2.5] μA
# Remember: synaptic weights need to be in μA/mV, so they're very small!
struct IFNeuron <: AbstractNeuronBlox
    params
    system
    namespace

	function IFNeuron(;name,
					   namespace=nothing, 
					   C=1.0,
					   θ = -50.0,
					   Eₘ= -70.0,
					   I_in=0)
		p = paramscoping(C=C, θ=θ, Eₘ=Eₘ, I_in=I_in)
		C, θ, Eₘ, I_in = p
		sts = @variables V(t)=-70.00 [output=true] jcn(t) [input=true]
		eqs = [D(V) ~ (I_in + jcn)/C]
		ev = [V~θ] => [V~Eₘ]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)

		new(p, sys, namespace)
	end
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

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- C: Membrane capicitance (μF).
- Eₘ: Resting membrane potential (mV).
- Rₘ: Membrane resistance (kΩ).
- τ: Synaptic time constant (ms).
- θ: Threshold voltage (mV).
- E_syn: Synaptic reversal potential (mV).
- G_syn: Synaptic conductance (μA/mV).
- I_in: External current input (μA).

References:
1. Sterratt, D., Graham, B., Gillies, A., & Willshaw, D. (2011). Principles of Computational Modelling in Neuroscience. Cambridge University Press.
"""
struct LIFNeuron <: AbstractNeuronBlox
    params
    system
    namespace
    # C = [1.0, 10.0] μF
    # Eₘ = [-100, -55] mV
    # Rₘ = [1, 100] kΩ
    # τ = [1.0, 100.0] ms
    # θ = [-65, -45] mV
    # E_syn = [-100, -55] mV
    # G_syn = [0.001, 0.01] μA/mV (bastardized μS - off by factor of 1000)
    # I_in = [-2.5, 2.5] μA (you will cook real neurons with these currents)
	function LIFNeuron(;name,
					   namespace=nothing, 
					   C=1.0,
					   Eₘ = -70.0,
					   Rₘ = 10.0,
					   τ = 10.0,
					   θ = -50.0,
					   E_syn=-70.0,
					   G_syn=0.002,
					   I_in=0.0)
		p = paramscoping(C=C, Eₘ=Eₘ, Rₘ=Rₘ, τ=τ, θ=θ, E_syn=E_syn, G_syn=G_syn, I_in=I_in)
		C, Eₘ, Rₘ, τ, θ, E_syn, G_syn, I_in = p
		sts = @variables V(t)=-70.00 G(t)=0.0 [output=true] jcn(t) [input=true]
		eqs = [ D(V) ~ (-(V-Eₘ)/Rₘ + I_in + jcn)/C,
				D(G)~(-1/τ)*G]

		ev = [V~θ] => [V~Eₘ, G~G+G_syn]
		sys = System(eqs, t, sts, p, continuous_events=[ev]; name=name)

		new(p, sys, namespace)
	end
end

struct LIFInhNeuron <: AbstractInhNeuronBlox
    system
    namespace

    function LIFInhNeuron(;
        name,
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
        inh_scaling_factor = 1 
    )

        ps = @parameters begin 
            g_L=g_L  
            V_L=V_L 
            V_E=V_E
            V_I=V_I
            V_reset=V_reset
            θ=θ
            C=C
            τ_AMPA=τ_AMPA 
            τ_GABA=τ_GABA 
            t_refract_duration=t_refract 
            t_refract_end=-Inf
            g_AMPA = g_AMPA * exci_scaling_factor
            g_AMPA_ext = g_AMPA_ext
            g_GABA = g_GABA * inh_scaling_factor
            g_NMDA = g_NMDA * exci_scaling_factor
            α=α
            Mg=Mg
            is_refractory=0
        end

        sts = @variables V(t)=-52 [output=true] S_AMPA(t)=0 S_GABA(t)=0 S_AMPA_ext(t)=0 jcn(t) [input=true] jcn_external(t) [input=true]
        eqs = [
            D(V) ~ (1 - is_refractory) * (- g_L * (V - V_L) - S_AMPA_ext * g_AMPA_ext * (V - V_E) - S_GABA * g_GABA * (V - V_I) - S_AMPA * g_AMPA * (V - V_E) - jcn) / C,
            D(S_AMPA) ~ - S_AMPA / τ_AMPA,
            D(S_GABA) ~ - S_GABA / τ_GABA,
            D(S_AMPA_ext) ~ - S_AMPA_ext / τ_AMPA
        ]

        refract_end = (t == t_refract_end) => [is_refractory ~ 0]

        sys = System(eqs, t, sts, ps; name=name, discrete_events = [refract_end])

		new(sys, namespace)
    end
end

struct LIFExciNeuron <: AbstractExciNeuronBlox
    system
    namespace

    function LIFExciNeuron(;
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
        inh_scaling_factor = 1 
    )

        ps = @parameters begin 
            g_L=g_L  
            V_L=V_L 
            V_E=V_E
            V_I=V_I
			V_reset=V_reset
            θ=θ
            C=C
            τ_AMPA=τ_AMPA 
            τ_GABA=τ_GABA 
            τ_NMDA_decay=τ_NMDA_decay 
            τ_NMDA_rise=τ_NMDA_rise 
            t_refract_duration=t_refract
            t_refract_end=-Inf
            g_AMPA = g_AMPA * exci_scaling_factor
            g_AMPA_ext = g_AMPA_ext
            g_GABA = g_GABA * inh_scaling_factor
            g_NMDA = g_NMDA * exci_scaling_factor
            α=α
            Mg=Mg
            is_refractory=0
        end

        sts = @variables V(t)=-52 [output=true] S_AMPA(t)=0 S_GABA(t)=0 S_NMDA(t)=0 x(t)=0 S_AMPA_ext(t)=0 jcn(t) [input=true] 
        eqs = [ 
            D(V) ~ (1 - is_refractory) * (- g_L * (V - V_L) - S_AMPA_ext * g_AMPA_ext * (V - V_E) - S_GABA * g_GABA * (V - V_I) - S_AMPA * g_AMPA * (V - V_E) - jcn) / C,
            D(S_AMPA) ~ - S_AMPA / τ_AMPA,
            D(S_GABA) ~ - S_GABA / τ_GABA,
            D(S_NMDA) ~ - S_NMDA / τ_NMDA_decay + α * x * (1 - S_NMDA),
            D(x) ~ - x / τ_NMDA_rise,
            D(S_AMPA_ext) ~ - S_AMPA_ext / τ_AMPA
        ]

        refract_end = (t == t_refract_end) => [is_refractory ~ 0]

        sys = System(eqs, t, sts, ps;  discrete_events = [refract_end], name=name)

		new(sys, namespace)
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
struct QIFNeuron <: AbstractNeuronBlox
    params
    system
    namespace

	function QIFNeuron(;name, 
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
						θ=25.0)
		p = paramscoping(C=C, Rₘ=Rₘ, E_syn=E_syn, G_syn=G_syn, τ₁=τ₁, τ₂=τ₂, I_in=I_in, Eₘ=Eₘ, Vᵣₑₛ=Vᵣₑₛ, θ=θ)
		C, Rₘ, E_syn, G_syn, τ₁, τ₂, I_in, Eₘ, Vᵣₑₛ, θ = p
		sts = @variables V(t)=-70.0 G(t)=0.0 [output=true] z(t)=0.0 jcn(t) [input=true]
		eqs = [ D(V) ~ ((V-Eₘ)^2/(Rₘ^2)+I_in+jcn)/C,
		 		D(G)~(-1/τ₂)*G + z,
	        	D(z)~(-1/τ₁)*z
	    	  ]
			  
		ev = (V > θ) => [V~Vᵣₑₛ,z~G_syn]
		sys = ODESystem(eqs, t, sts, p, discrete_events=[ev]; name=name)

		new(p, sys, namespace)
	end
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
struct IzhikevichNeuron <: AbstractNeuronBlox
    params
    system
    namespace

	function IzhikevichNeuron(;name,
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
							   τ=2.6)
		p = paramscoping(α=α, η=η, a=a, b=b, θ=θ, vᵣ=vᵣ, wⱼ=wⱼ, sⱼ=sⱼ, gₛ=gₛ, eᵣ=eᵣ, τ=τ)
		α, η, a, b, θ, vᵣ, wⱼ, sⱼ, gₛ, eᵣ, τ = p
		sts = @variables V(t)=0.0 w(t)=0.0 G(t)=0.0 [output=true] z(t)=0.0 jcn(t) [input=true]
		eqs = [ D(V) ~ V*(V-α) - w + η + jcn,
				D(w) ~ a*(b*V - w),
				D(G) ~ (-1/τ)*G + z,
				D(z) ~ (-1/τ)*z
			  ]
		ev = [V~θ] => [V~vᵣ, w~w+wⱼ, z~sⱼ]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)

		new(p, sys, namespace)
	end
end

# ====================
# Metabolic HH Neuron
# ====================

# Labels for excitatory vs inhibitory neuron subtype (determines synaptic parameters)
const Excitatory = :Excitatory
const Inhibitory = :Inhibitory

"""
	MetabolicHHNeuron(name, namespace, neurontype,
		Naᵢᵧ, ρₘₐₓ, α, λ, ϵ₀, O₂ᵦ, γ, β, ϵₖ, Kₒᵦ, Gᵧ, Clᵢ, Clₒ, R, T, F,
		Gₙₐ, Gₖ, Gₙₐ_L, Gₖ_L, G_cl_L, C_m, I_in, G_exc, G_inh, E_syn_exc, E_syn_inh)

	Create a Metabolic Hodgkin-Huxley Neuron. This model accounts for
	dynamic ion concentrations, oxygen consumption and astrocytic buffering.

Arguments:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- neurontype: excitatory or inhibitory.
- Naᵢᵧ: Intracellular Na+ concentration (mM).
- ρₘₐₓ: Maximum pump rate (mM/s).
- α: Conversion factor from pump current to O2 consumption rate (g/mol).
- λ: Relative cell density.
- ϵ₀: O2 diffusion rate (s^-1).
- O₂ᵦ: O2 buffer concentration (mg/L).
- γ: Conversion factor from current to concentration (mM/s)/(uA/cm2).
- β: Ratio of intracellular vs extracellular volume.
- ϵₖ: K+ diffusion rate (1/s).
- Kₒᵦ: K+ buffer concentration (mM).
- Gᵧ: Glia uptake strength of K+ (mM/s).
- Clᵢ: Intracellular Cl- concentration (mM).
- Clₒ: Extracellular Cl- concentration (mM).
- R: Ideal gas constant (J/(mol*K)).
- T: Temperature (K).
- F: Faraday's constant (C/mol).
- Gₙₐ: Na+ maximum conductance (mS/cm^2).
- Gₖ: K+ maximum conductance (mS/cm^2).
- Gₙₐ_L: Na+ leak conductance (mS/cm^2).
- Gₖ_L: K+ leak conductance (mS/cm^2).
- G_cl_L: Cl- leak conductance (mS/cm^2).
- C_m: Membrane capacitance (uF/cm^2).
- I_in: External current input (uA/cm^2).
- G_exc: Conductance of excitatory synapses (mS/cm^2).
- G_inh: Conductance of inhibitory synapses (mS/cm^2).
- E_syn_exc: Excitatory synaptic reversal potential (mV).
- E_syn_inh: Inhibitory synaptic reversal potential (mV).

References:
1. Dutta, Shrey, et al. "Mechanisms underlying pathological cortical bursts during metabolic depletion." Nature Communications 14.1 (2023): 4792.

"""
struct MetabolicHHNeuron{IsExcitatory} <: AbstractNeuronBlox
	system
    output
    namespace

	function MetabolicHHNeuron(
		;name,
		namespace=nothing,
		neurontype=:excitatory,
		Naᵢᵧ = 18.0,  # Intracellular Naconcentration, in mM
		ρₘₐₓ = 1.25,  # Maximum pump rate, in mM/s
		α = 5.3,  # Conversion factor from pump current to O2 consumption rate, in g/mol
		λ = 1.,  # Relative cell density [!]
		ϵ₀ = 0.17,  # O2 diffusion rate, in s^-1
		O₂ᵦ = 32.,  # O2 buffer concentration, in mg/L
		γ = 0.0445,  # conversion factor from current to concentration, in (mM/s)/(uA/cm2)
		β = 7.,  # Ratio of intracellular vs extracellular volume
		ϵₖ = 0.33,  # K+ diffusion rate, in 1/s
		Kₒᵦ = 3.5,  # K+ buffer concentration, in mM
		Gᵧ = 8.0,  # Glia uptake strength of K+, in mM/s
		Clᵢ = 6.0, # Intracellular Cl- concentration, in mM
		Clₒ = 130.0, # Extracellular Cl- concentration, in mM
		R = 8.314,  # Ideal gas constant, in J/(mol*K)
		T = 310.0,  # Temperature, in K
		F = 96485.0,  # Faraday's constant, in C/mol
		Gₙₐ = 30.,  # Na+ maximum conductance, in mS/cm^2
		Gₖ = 25.,  # K+ maximum conductance, in mS/cm^2
		Gₙₐ_L = 0.0175,  # Na+ leak conductance, in mS/cm^2
		Gₖ_L = 0.05,  # K+ leak conductance, in mS/cm^2
		G_cl_L = 0.05,  # Cl- leak conductance, in mS/cm^2
		C_m = 1.,  # Membrane capacitance, in uF/cm^2
		I_in = 0.,  # External current input, in uA/cm^2
		G_exc = 0.022,  # Conductance of excitatory synapses, in mS/cm^2
		G_inh = 0.374,  # Conductance of inhibitory synapses, in mS/cm^2
		E_syn_exc = 0., # Excitatory synaptic reversal potential, in mV
		E_syn_inh = -80.,  # Inhibitory synaptic reversal potential, in mV
		τ = 4.,  # Time constant for synapse, in ms [!]
		)
		if neurontype == :excitatory
			MetabolicHHNeuron{Excitatory}(;name, namespace,
				Naᵢᵧ, ρₘₐₓ, α, λ, ϵ₀, O₂ᵦ, γ, β, ϵₖ, Kₒᵦ, Gᵧ, Clᵢ, Clₒ, R, T, F,
				Gₙₐ, Gₖ, Gₙₐ_L, Gₖ_L, G_cl_L, C_m, I_in, G=G_exc, E_syn=E_syn_exc, τ
			)
		elseif neurontype == :inhibitory
			MetabolicHHNeuron{Inhibitory}(;name, namespace,
				Naᵢᵧ, ρₘₐₓ, α, λ, ϵ₀, O₂ᵦ, γ, β, ϵₖ, Kₒᵦ, Gᵧ, Clᵢ, Clₒ, R, T, F,
				Gₙₐ, Gₖ, Gₙₐ_L, Gₖ_L, G_cl_L, C_m, I_in, G=G_inh, E_syn=E_syn_inh, τ
			)
		else
			error("Unknown neuron type: $neurontype")
		end
	end
	function MetabolicHHNeuron{Excitatory}(
		;name,
		namespace=nothing,
		Naᵢᵧ, ρₘₐₓ, α, λ, ϵ₀, O₂ᵦ, γ, β, ϵₖ, Kₒᵦ, Gᵧ, Clᵢ, Clₒ, R, T, F,
		Gₙₐ, Gₖ, Gₙₐ_L, Gₖ_L, G_cl_L, C_m, I_in, G, E_syn, τ
		)
		# Parameters
		ps = @parameters begin
			Naᵢᵧ=Naᵢᵧ
			ρₘₐₓ=ρₘₐₓ
			α=α
			λ=λ
			ϵ₀=ϵ₀
			O₂ᵦ=O₂ᵦ
			γ=γ
			β=β
			ϵₖ=ϵₖ
			Kₒᵦ=Kₒᵦ
			Gᵧ=Gᵧ
			R=R
			T=T
			F=F
			Gₙₐ=Gₙₐ
			Gₖ=Gₖ
			Gₙₐ_L=Gₙₐ_L
			Gₖ_L=Gₖ_L
			G_cl_L=G_cl_L
			C_m=C_m
			I_in=I_in
			G=G
			E_syn=E_syn
			τ=τ
		end
		# State variables
		sts = @variables begin
			V(t)=-60.0
			O₂(t)=25.0
			Kₒ(t)=3.0
			Naᵢ(t)=15.0
			m(t)=0.0
			h(t)=0.0
			n(t)=0.0
			I_syn(t) 
			[input=true] 
			S(t)=0.1
			[output = true] 
			χ(t)=0.0
			[output = true] 
		end
		
		# Pump currents
		ρ = ρₘₐₓ / (1.0 + exp((20.0 - O₂)/3.0))
		I_pump = ρ / (1.0 + exp((25.0 - Naᵢ)/3.0)*(1.0 + exp(5.5 - Kₒ)))
		I_gliapump = ρ / (3.0*(1.0 + exp((25.0 - Naᵢᵧ)/3.0))*(1.0 + exp(5.5 - Kₒ)))

		# Glia current
		I_glia = Gᵧ / (1.0 + exp((18.0 - Kₒ)/2.5))

		# Ion concentrations
		Kᵢ = 140.0 + (18.0 - Naᵢ)
		Naₒ = 144.0 - β*(Naᵢ - 18.0)
	
		# Ion reversal potentials
		Eₙₐ = R*T/F * log(Naₒ/Naᵢ) * 1000.0
		Eₖ = R*T/F * log(Kₒ/Kᵢ) * 1000.0
		E_cl = R*T/F * log(Clᵢ/Clₒ) * 1000.0
		
		# Ion currents
		Iₙₐ = Gₙₐ*m^3.0*h*(V - Eₙₐ) + Gₙₐ_L*(V - Eₙₐ)
		Iₖ = Gₖ*n^4.0*(V - Eₖ) + Gₖ_L*(V - Eₖ)
		I_cl = G_cl_L*(V - E_cl)

		# Ion channel gating rate equations
		aₘ = 0.32*(V + 54.0)/(1.0 - exp(-0.25*(V + 54.0)))
		bₘ = 0.28*(V + 27.0)/(exp(0.2*(V + 27.0)) - 1.0)
		aₕ = 0.128*exp(-(V + 50.0)/18.0)
		bₕ = 4.0/(1.0 + exp(-0.2*(V + 27.0)))
		aₙ = 0.032*(V + 52.0)/(1.0 - exp(-0.2*(V + 52.0)))
		bₙ = 0.5*exp(-(V + 57.0)/40.0)
		
		# Depolarization factor, as continuous variable
		η = 0.4/(1.0 + exp(-10.0*(V + 30.0)))/(1.0 + exp(10.0*(V + 10.0)))

		# Differential equations
		eqs = [
			D(O₂) ~ -α*λ*(I_pump + I_gliapump) + ϵ₀*(O₂ᵦ - O₂),
			D(Kₒ) ~ γ*β*Iₖ - 2.0*β*I_pump - I_glia - 2.0*I_gliapump - ϵₖ*(Kₒ - Kₒᵦ),
			D(Naᵢ) ~ -γ*Iₙₐ - 3.0*I_pump,
			D(m) ~ aₘ * (1.0 - m) - bₘ*m,
			D(h) ~ aₕ * (1.0 - h) - bₕ*h,
			D(n) ~ aₙ * (1.0 - n) - bₙ*n,
			D(V) ~ (-Iₙₐ - Iₖ - I_cl - I_syn - I_in)/C_m,
			D(S) ~ (20.0/(1.0 + exp(-(V + 20.0)/3.0)) * (1.0 - S) - S)/τ,
			D(χ) ~ η*(V + 50.0) - 0.4*χ
			]

		# Define the ODE system
		sys = ODESystem(eqs, t, sts, ps; name=name)

		# Construct the neuron
		new{Excitatory}(sys, sts[1], namespace)
	end

	function MetabolicHHNeuron{Inhibitory}(
		;name,
		namespace=nothing,
		Naᵢᵧ, ρₘₐₓ, α, λ, ϵ₀, O₂ᵦ, γ, β, ϵₖ, Kₒᵦ, Gᵧ, Clᵢ, Clₒ, R, T, F,
		Gₙₐ, Gₖ, Gₙₐ_L, Gₖ_L, G_cl_L, C_m, I_in, G, E_syn, τ
		)
		# Parameters
		ps = @parameters begin
			Naᵢᵧ=Naᵢᵧ
			ρₘₐₓ=ρₘₐₓ
			α=α
			λ=λ
			ϵ₀=ϵ₀
			O₂ᵦ=O₂ᵦ
			γ=γ
			β=β
			ϵₖ=ϵₖ
			Kₒᵦ=Kₒᵦ
			Gᵧ=Gᵧ
			R=R
			T=T
			F=F
			Gₙₐ=Gₙₐ
			Gₖ=Gₖ
			Gₙₐ_L=Gₙₐ_L
			Gₖ_L=Gₖ_L
			G_cl_L=G_cl_L
			C_m=C_m
			I_in=I_in
			G=G
			E_syn=E_syn
			τ=τ
		end
		# State variables
		sts = @variables begin
			V(t)=-60.0
			O₂(t)=25.0
			Kₒ(t)=3.0
			Naᵢ(t)=15.0
			m(t)=0.0
			h(t)=0.0
			n(t)=0.0
			I_syn(t) 
			[input=true] 
			S(t)=0.1
			[output = true] 
			χ(t)=0.0
			[output = true] 
		end
		
		# Pump currents
		ρ = ρₘₐₓ / (1.0 + exp((20.0 - O₂)/3.0))
		I_pump = ρ / (1.0 + exp((25.0 - Naᵢ)/3.0)*(1.0 + exp(5.5 - Kₒ)))
		I_gliapump = ρ / (3.0*(1.0 + exp((25.0 - Naᵢᵧ)/3.0))*(1.0 + exp(5.5 - Kₒ)))

		# Glia current
		I_glia = Gᵧ / (1.0 + exp((18.0 - Kₒ)/2.5))

		# Ion concentrations
		Kᵢ = 140.0 + (18.0 - Naᵢ)
		Naₒ = 144.0 - β*(Naᵢ - 18.0)
	
		# Ion reversal potentials
		Eₙₐ = R*T/F * log(Naₒ/Naᵢ) * 1000.0
		Eₖ = R*T/F * log(Kₒ/Kᵢ) * 1000.0
		E_cl = R*T/F * log(Clᵢ/Clₒ) * 1000.0
		
		# Ion currents
		Iₙₐ = Gₙₐ*m^3.0*h*(V - Eₙₐ) + Gₙₐ_L*(V - Eₙₐ)
		Iₖ = Gₖ*n^4.0*(V - Eₖ) + Gₖ_L*(V - Eₖ)
		I_cl = G_cl_L*(V - E_cl)

		# Ion channel gating rate equations
		aₘ = 0.32*(V + 54.0)/(1.0 - exp(-0.25*(V + 54.0)))
		bₘ = 0.28*(V + 27.0)/(exp(0.2*(V + 27.0)) - 1.0)
		aₕ = 0.128*exp(-(V + 50.0)/18.0)
		bₕ = 4.0/(1.0 + exp(-0.2*(V + 27.0)))
		aₙ = 0.032*(V + 52.0)/(1.0 - exp(-0.2*(V + 52.0)))
		bₙ = 0.5*exp(-(V + 57.0)/40.0)
		
		# Depolarization factor, as continuous variable
		η = 0.4/(1.0 + exp(-10.0*(V + 30.0)))/(1.0 + exp(10.0*(V + 10.0)))

		# Differential equations
		eqs = [
			D(O₂) ~ -α*λ*(I_pump + I_gliapump) + ϵ₀*(O₂ᵦ - O₂),
			D(Kₒ) ~ γ*β*Iₖ - 2.0*β*I_pump - I_glia - 2.0*I_gliapump - ϵₖ*(Kₒ - Kₒᵦ),
			D(Naᵢ) ~ -γ*Iₙₐ - 3.0*I_pump,
			D(m) ~ aₘ * (1.0 - m) - bₘ*m,
			D(h) ~ aₕ * (1.0 - h) - bₕ*h,
			D(n) ~ aₙ * (1.0 - n) - bₙ*n,
			D(V) ~ (-Iₙₐ - Iₖ - I_cl - I_syn - I_in)/C_m,
			D(S) ~ (20.0/(1.0 + exp(-(V + 20.0)/3.0)) * (1.0 - S) - S)/τ,
			D(χ) ~ η*(V + 50.0) - 0.4*χ
			]

		# Define the ODE system
		sys = ODESystem(eqs, t, sts, ps; name=name)

		# Construct the neuron
		new{Inhibitory}(sys, sts[1], namespace)
	end
end
