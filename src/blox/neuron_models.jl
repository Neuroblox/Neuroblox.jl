abstract type AbstractInhNeuronBlox <: AbstractNeuronBlox end
abstract type AbstractExciNeuronBlox <: AbstractNeuronBlox end
@parameters t
D = Differential(t)

struct HHNeuronExciBlox <: AbstractExciNeuronBlox
    odesystem
    output
    namespace

	function HHNeuronExciBlox(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        G_syn=3, 
        I_bg=0,
        freq=0,
        phase=0,
        τ=5
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
			[input=true]
			G(t)=0.0 
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

		new(sys, spikes, namespace)
	end
end	

struct HHNeuronInhibBlox <: AbstractInhNeuronBlox
    odesystem
    namespace
	function HHNeuronInhibBlox(;
        name, 
        namespace = nothing, 
        E_syn=-70.0,
        G_syn=11.5,
        I_bg=0,
        freq=0,
        phase=0,
        τ=70
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)=0.0 
			[input=true] 
			I_asc(t)=0.0
			[input=true]
			I_in(t)=0.0
			[input=true]
            G(t)=0.0 
			[output = true] 
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
    output
    jcn
	voltage
    odesystem
    namespace
	function IFNeuron(;name,
					   namespace=nothing, 
					   C=1.0,
					   θ = -50.0,
					   Eₘ= -70.0,
					   I_in=0)
		p = paramscoping(C=C, θ=θ, Eₘ=Eₘ, I_in=I_in)
		C, θ, Eₘ, I_in = p
		sts = @variables V(t) = -70.00 jcn(t)=0.0 [input=true]
		eqs = [D(V) ~ (I_in + jcn)/C]
		ev = [V~θ] => [V~Eₘ]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)
		new(p, sts[1], sts[2], sts[1], sys, namespace)
	end
end

# Paramater bounds for GUI
# C = [0.1, 100] μF
# Eₘ = [-100, -55] mV
# Rₘ = [50, 200] Ω
# τ₁ = [0.01, 1] ms
# τ₂ = [1.0, 10.0] ms
# τᵣ = [1, 10] ms
# θ = [-65, -45] mV
# E_syn = [-10, 10]
# G_syn = [0.1, 1]
# I_in = [-2.0, 2.0] mA
# freq = [0, 100]
# phase = [0, 2π]
# struct LIFNeuron <: AbstractNeuronBlox
# 	params
#     output
#     jcn
# 	voltage
#     odesystem
#     namespace
# 	function LIFNeuron(;name,
# 					   namespace=nothing, 
# 					   C=1.0,
# 					   Eₘ = -70.0,
# 					   Rₘ = 100.0,
# 					   τ₁=0.1,
# 					   τ₂=10.0,
# 					   τᵣ=3,
# 					   θ = -50.0,
# 					   E_syn=0,
# 					   G_syn=0.2,
# 					   I_in=0,
# 					   freq=0,
# 					   phase=0)
# 		p = paramscoping(C=C, Eₘ=Eₘ, Rₘ=Rₘ, τ₁=τ₁, τ₂=τ₂, τᵣ=τᵣ, θ=θ, E_syn=E_syn, G_syn=G_syn, I_in=I_in, freq=freq, phase=phase)
# 		C, Eₘ, Rₘ, τ₁, τ₂, τᵣ, θ, E_syn, G_syn, I_in, freq, phase = p
# 		sts = @variables V(t) = -70.00 G(t)=0.0 z(t)=0.0 Cₜ(t) = 0.0 jcn(t)=0.0 [input=true]
# 		eqs = [ D(V) ~ (-(V-Eₘ)/Rₘ + I_in*(sin((t*freq*2*pi/1000)+phase)+1) + jcn)/(C+Cₜ),
# 				D(G)~(-1/τ₂)*G + z,
# 				D(z)~(-1/τ₁)*z,
# 				D(Cₜ)~(-1/τᵣ)*Cₜ
# 			  ]
# 		ev = [V~θ] => [V~Eₘ, z~G_syn,Cₜ~10]
# 		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)
# 		new(p, sts[2], sts[5], sts[1], sys, namespace)
# 	end
# end


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
# C = [1.0, 10.0] μF
# Eₘ = [-100, -55] mV
# Rₘ = [1, 100] kΩ
# τ = [1.0, 100.0] ms
# θ = [-65, -45] mV
# E_syn = [-100, -55] mV
# G_syn = [0.001, 0.01] μA/mV (bastardized μS - off by factor of 1000)
# I_in = [-2.5, 2.5] μA (you will cook real neurons with these currents)
struct LIFNeuron <: AbstractNeuronBlox
	params
    output
    jcn
	voltage
    odesystem
    namespace
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
		sts = @variables V(t) = -70.00 G(t)=0.0 z(t)=0.0 Cₜ(t) = 0.0 jcn(t)=0.0 [input=true]
		eqs = [ D(V) ~ (-(V-Eₘ)/Rₘ + I_in + jcn)/C,
				D(G)~(-1/τ)*G,
			  ]

		ev = [V~θ] => [V~Eₘ, G~G+G_syn]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)
		new(p, sts[2], sts[5], sts[1], sys, namespace)
	end
end

# Paramater bounds for GUI
# C = [0.1, 100]
# ω = [0, 100]
# E_syn = [-10, 10]
# G_syn = [0.1, 1]
# τ₁ = [1, 100]
# τ₂ = [1, 100]
# I_in = [-25, 25]
# Eₘ = [-10, 10]
# Vᵣₑₛ = [-100, -55]
# θ = [0, 50]

struct QIFNeuron <: AbstractNeuronBlox
	params
    output
    jcn
	voltage
    odesystem
    namespace
	function QIFNeuron(;name, 
						namespace=nothing,
						C=1.0,
						ω=0.0,
						E_syn=0.0,
						G_syn=1.0, 
						τ₁=10.0,
						τ₂=10.0,
						I_in=0.0, 
						Eₘ=0.0,
						Vᵣₑₛ=-70.0,
						θ=25.0)
		p = paramscoping(C=C, ω=ω, E_syn=E_syn, G_syn=G_syn, τ₁=τ₁, τ₂=τ₂, I_in=I_in, Eₘ=Eₘ, Vᵣₑₛ=Vᵣₑₛ, θ=θ)
		C, ω, E_syn, G_syn, τ₁, τ₂, I_in, Eₘ, Vᵣₑₛ, θ = p
		sts = @variables V(t) = -70.0 G(t)=0.0 z(t)=0.0 jcn(t)=0.0 [input=true]
		eqs = [ D(V) ~ ((V-Eₘ)^2+I_in+jcn)/C,
		 		D(G)~(-1/τ₂)*G + z,
	        	D(z)~(-1/τ₁)*z
	    	  ]
   		ev = [V~θ] => [V~Vᵣₑₛ,z~G_syn]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)
		new(p, sts[2], sts[4], sts[1], sys, namespace)
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
    output
    jcn
	voltage
    odesystem
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
		sts = @variables V(t)=0.0 w(t)=0.0 G(t)=0.0 z(t)=0.0 jcn(t)=0.0 [input=true]
		eqs = [ D(V) ~ V*(V-α) - w + η + jcn,
				D(w) ~ a*(b*V - w),
				D(G) ~ (-1/τ)*G + z,
				D(z) ~ (-1/τ)*z
			  ]
		ev = [V~θ] => [V~vᵣ, w~w+wⱼ, z~sⱼ]
		sys = ODESystem(eqs, t, sts, p, continuous_events=[ev]; name=name)
		new(p, sts[2], sts[5], sts[1], sys, namespace)
	end
end