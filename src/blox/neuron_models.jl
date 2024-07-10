abstract type AbstractInhNeuronBlox <: AbstractNeuronBlox end
abstract type AbstractExciNeuronBlox <: AbstractNeuronBlox end

#These HH neuron models were used in Pathak et al 2023 model of corticostriatal circuit
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

#These neurons were used in Adam et. al 2022 model for DBS

struct HHNeuronInhib_MSN_Adam_Blox <: AbstractInhNeuronBlox
    odesystem
    namespace

	function HHNeuronInhib_MSN_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=1.19,
        freq=0,
        phase=0,
        τ=13,
        Cₘ=1.0,
		σ=4.0,
		a=2,
		b=4,
		T=37
    )
		sts = @variables begin 
			V(t)=-63.83 
			n(t)=0.062
			m(t)=0.027
			h(t)=0.99
			mM(t)=0.022
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
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
    odesystem
    namespace

	function HHNeuronInhib_FSI_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=5.5,
        freq=0,
        phase=0,
        τ=11,
		τₛ=6.5,
        Cₘ=1.0,
		σ=60.0,
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
			I_syn(t)=0.0 
			[input=true] 
			I_gap(t)=0.0
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
			[input=true]
			G(t)=0.0 
			[output = true] 
			Gₛ(t)=0.0 
			[output = true] 
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
    odesystem
    namespace

	function HHNeuronExci_STN_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        I_bg=1.9,
        freq=0,
        phase=0,
        τ=2,
        Cₘ=1.0,
		σ=80.0,
		a=5,
		b=4
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
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

struct HHNeuronInhib_GPe_Adam_Blox <: AbstractInhNeuronBlox
    odesystem
    namespace

	function HHNeuronInhib_GPe_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=3.0,
        freq=0,
        phase=0,
        τ=10,
        Cₘ=1.0,
		σ=80.0,
		a=2,
		b=4,
		T=37
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
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

# The following are general HH neurons used to model pyramidal and interneurons in Adam et al 2024 model for Ketamine affect on cortex
struct HHNeuronExci_pyr_Adam_Blox <: AbstractExciNeuronBlox
    odesystem
    namespace

	function HHNeuronExci_pyr_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        I_bg=-0.25,
        freq=0,
        phase=0,
        τ=1.5,
		τ_glu=1.2,
        Cₘ=1.0,
		σ=20.0,
		a=5,
		b=4
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
			[input=true]
			G(t)=0.0 
			[output = true] 
			Glu(t)=0.0
			[output = true] 
		end

		ps = @parameters begin 
			E_syn=E_syn 
			G_Na = 100 
			G_K  = 80 
			G_L = 0.05 
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
		
		G_asymp(v,a,b) = a*(1+tanh(v/b))
		
		eqs = [ 
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in+σ*χ), 
			   D(n)~(αₙ(V)*(1-n)-βₙ(V)*n), 
			   D(m)~(αₘ(V)*(1-m)-βₘ(V)*m), 
			   D(h)~(αₕ(V)*(1-h)-βₕ(V)*h),
			   D(G)~(-1/τ)*G + G_asymp(V,a,b)*(1-G),
			   D(Glu)~(-1/τ_glu)*Glu + G_asymp(V-20,2.35,0.01) #this approximates the glutamate dynamics in the model where every spike instantaneously raises glutamate to 1mM
			  
		]
        
		sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)

		new(sys, namespace)
	end
end	

struct HHNeuronInh_inter_Adam_Blox <: AbstractInhNeuronBlox
    odesystem
    namespace

	function HHNeuronInh_inter_Adam_Blox(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=0.1,
        freq=0,
        phase=0,
        τ=6,
        Cₘ=1.0,
		σ=20.0,
		a=2,
		b=4
    )
		sts = @variables begin 
			V(t)=-67.00 
			n(t)=0.032 
			m(t)=0.05 
			h(t)=0.059 
			I_syn(t)=0.0 
			[input=true] 
            I_in(t)=0.0
            [input=true]
			I_asc(t)=0.0
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
    output
    jcn
	voltage
    odesystem
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
		sts = @variables V(t) = -70.0 G(t)=0.0 z(t)=0.0 jcn(t)=0.0 [input=true]
		eqs = [ D(V) ~ ((V-Eₘ)^2/(Rₘ^2)+I_in+jcn)/C,
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