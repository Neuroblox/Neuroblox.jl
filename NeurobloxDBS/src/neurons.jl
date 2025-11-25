struct HHNeuronInhib_MSN_Adam <: AbstractInhNeuron
    system
    namespace

	function HHNeuronInhib_MSN_Adam(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=1.172,
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
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)-G_M*mM*(V-E_K)+I_bg+I_syn+I_asc+I_in+σ*χ), 
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

struct HHNeuronInhib_FSI_Adam <: AbstractInhNeuron
    system
    namespace

	function HHNeuronInhib_FSI_Adam(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=6.2,
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
			   D(V)~(1/Cₘ)*(-G_Na*m_inf(V)^3*h*(V-E_Na)-G_K*n^2*(V-E_K)-G_L*(V-E_L)-G_D*mD^3*hD*(V-E_K)+I_bg+I_syn+I_gap+I_asc+I_in+σ*χ), 
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

struct HHNeuronExci_STN_Adam <: AbstractExciNeuron
    system
    namespace

	function HHNeuronExci_STN_Adam(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        I_bg=1.8,
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
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+σ*χ), 
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

struct HHNeuronInhib_GPe_Adam <: AbstractInhNeuron
    system
    namespace

	function HHNeuronInhib_GPe_Adam(;
        name, 
        namespace=nothing,
        E_syn=-80.0, 
        I_bg=3.4,
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
			   D(V)~(1/Cₘ)*(-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+σ*χ), 
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

