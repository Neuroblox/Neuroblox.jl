abstract type AbstractInhNeuronBlox <: AbstractNeuronBlox end
abstract type AbstractExciNeuronBlox <: AbstractNeuronBlox end
@parameters t
D = Differential(t)

#Quadratic Integrate and Fire neurons 
mutable struct QIFNeuronBlox <: AbstractNeuronBlox
    # all parameters are Num as to allow symbolic expressions
    C::Num
    E_syn::Num
    G_syn::Num
    ω::Num
	τ::Num
    connector::Num
	noDetail::Vector{Num}
    detail::Vector{Num}
	initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
	function QIFNeuronBlox(;name,C=1.0,E_syn=0, G_syn=1,ω=0,τ=10)

    	sts = @variables V(t) = -70.0 G(t)=0.0 z(t)=0.0 I_syn(t)=0.0 jcn(t)=0.0
		ps = @parameters C=C ω=ω I_in=(ω*C/2)^2 Eₘ=0.0 Vᵣₑₛ=-70.0 θ=25 τ₁=τ τ₂=τ E_syn=E_syn G_syn=G_syn
	
		eqs = [
		 	D(V) ~ ((V-Eₘ)^2+I_in+I_syn)/C,
		 	D(G)~(-1/τ₂)*G + z,
	        D(z)~(-1/τ₁)*z
	    ]
   		ev = [V~θ] => [V~Vᵣₑₛ,z~G_syn]
		odesys = ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
		new(C, E_syn, G_syn, ω, τ, odesys.G, 
		[odesys.V],[odesys.V, odesys.G],Dict{Num, Tuple{Float64, Float64}}(),odesys)
	end
end

"""
thetaneuron has the following parameters:
    η:     Constant drive
    α_inv: Time to peak of spike
    k:     All-to-all coupling strength
and the following variables:
    θ(t):   Theta neuron state
    g(t):   Synaptic current
    jcn(t): Synaptic input
and returns:
    an ODE System
"""
function theta_neuron(;name, η=η, α_inv=α_inv, k=k)

    params = @parameters η=η α_inv=α_inv k=k
    sts    = @variables θ(t)=0.0 g(t)=0.0 jcn(t)=0.0
    
    eqs = [D(θ) ~ 1-cos(θ) + (1+cos(θ))*(η + k*g),
          D(g) ~ α_inv*(jcn - g)]

    return ODESystem(eqs, t, sts, params; name=name)

end

# Leaky Integrate and Fire neuron with synaptic dynamics
mutable struct IFNeuronBlox <: AbstractNeuronBlox
    # all parameters are Num as to allow symbolic expressions
    C::Num
    E_syn::Num
    G_syn::Num
	I_in::Num
	freq::Num
	phase::Num
	τ::Num
    connector::Num
	noDetail::Vector{Num}
    detail::Vector{Num}
	initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
	function IFNeuronBlox(;name,C=1.0,E_syn=0,G_syn=0.2,I_in=0,freq=0,phase=0,τ=10)

		sts = @variables V(t) = -70.00 G(t)=0.0 z(t)=0.0 spt(t)=0.0 Cₜ(t) = 0.0 I_syn(t)=0 jcn(t)=0.0
		ps = @parameters C=C I_in = I_in Eₘ = -70.0 Rₘ = 100.0 θ = -50.0 τ₁=0.1 τ₂=τ E_syn=E_syn G_syn=G_syn phase=phase τᵣ=3

		eqs = [
		    	D(V) ~ (-(V-Eₘ)/Rₘ + I_in*(sin((t*freq*2*pi/1000)+phase)+1) + I_syn)/(C+Cₜ),
		    	D(G)~(-1/τ₂)*G + z,
	        	D(z)~(-1/τ₁)*z,
				D(spt)~0,
		    	D(Cₜ)~(-1/τᵣ)*Cₜ
		  	]
    		ev = [V~θ] => [V~Eₘ,z~G_syn,Cₜ~10]
		odesys = ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
		new(C, E_syn, G_syn, I_in, freq, phase, τ, odesys.G,
		[odesys.V],[odesys.V, odesys.G],Dict{Num, Tuple{Float64, Float64}}(),odesys)
	end
end

"""
Standard Leaky Integrate and Fire neuron model.

variables:
    V(t):  Membrane voltage
    jcn:   Input from other neurons
parameters:
    I_in:   Input current
	V_L:    Resting state potential
    τ:      Membrane timescale
	R:      Membrane resistance
	θ:      Spike threshold
	st:     Last spike time
	strain: Spike train
returns:
    an ODE System
"""
mutable struct LIFNeuronBlox <: AbstractComponent
	I_in::Num
	V_L::Num
	τ::Num
	R::Num
	θ::Num
    connector::Num
	noDetail::Vector{Num}
    detail::Vector{Num}
	initial::Dict{Num, Tuple{Float64, Float64}}
    odesystem::ODESystem
	function LIFNeuronBlox(;name, I_in=0, V_L=-70.0, τ=10.0, R=100.0, θ = -10.0)
		sts = @variables V(t) = -70.0 jcn(t) = 0.0
		par = @parameters I_in=I_in V_L=V_L R=R τ=τ st=-Inf strain=[]
		eqs = [
		    	D(V) ~ (-V + V_L + R*(I_in + jcn))/τ
		  	  ]

		function lif_affect!(integ, u, p, ctx)
			integ.u[u.V] = integ.p[p.V_L]
			integ.p[p.st] = integ.t
			push!(integ.p[p.strain], integ.t)
		end

    	spike = [V ~ θ] => (lif_affect!, [V], [V_L, st, strain], nothing)

		odesys = ODESystem(eqs, t, sts, par; continuous_events=spike, name=name)
		new(I_in, V_L, τ, R, θ, odesys.V, [odesys.V],[odesys.V],Dict{Num, Tuple{Float64, Float64}}(),odesys)
	end
end

function spike_affect!(integ, u, p, ctx)
    du = SciMLBase.get_du(integ)
    if du[u.V] > 0
        integ.u[u.spikes_cumulative] += 1
        integ.u[u.spikes_window] += 1
    end
end

struct HHNeuronExciBlox <: AbstractExciNeuronBlox
    odesystem
    output
    namespace

	function HHNeuronExciBlox(;
        name, 
        namespace=nothing,
        t_spike_window=90.0,
        θ_spike=0.0,
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
               # HACK : need to define a Differential equation for spikes
               # the alternative of having it as an algebraic equation with [irreducible=true]
               # leads to incorrect or unstable solutions. Needs more attention!
               D(spikes_cumulative) ~ spk_const*G_asymp(V,G_syn),
               D(spikes_window) ~ spk_const*G_asymp(V,G_syn)
		]
        
        spike_reset_cb = [(t_spike_window + eps(float(t_spike_window))) => [spikes_window ~ 0]]

		sys = ODESystem(
            eqs, t, sts, ps; 
            name = Symbol(name),discrete_events = spike_reset_cb
			#name = Symbol(name)
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
        t_spike_window=90.0,
        θ_spike=0.0,
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
			   D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
               D(spikes_cumulative) ~ spk_const*G_asymp(V,G_syn),
               D(spikes_window) ~ spk_const*G_asymp(V,G_syn)
		]

        spike_reset_cb = [(t_spike_window + eps(float(t_spike_window))) => [spikes_window ~ 0]]

        sys = ODESystem(
            eqs, t, sts, ps; 
            name = Symbol(name), discrete_events = spike_reset_cb
			#name = Symbol(name)
        )
        
		new(sys, namespace)
	end
end	

	



	


