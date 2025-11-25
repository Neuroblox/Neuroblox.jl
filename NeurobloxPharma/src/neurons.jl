struct HHNeuronExci <: AbstractExciNeuron
    system
    namespace

	function HHNeuronExci(;
        name, 
        namespace=nothing,
        E_syn=0.0, 
        G_syn=3, 
        I_bg=0.0,
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
			jcn(t) 
			[input=true]
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
			   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn, 
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

struct HHNeuronInhib <: AbstractInhNeuron
    system
    namespace
	function HHNeuronInhib(;
        name, 
        namespace = nothing, 
        E_syn=-70.0,
        G_syn=11.5,
        I_bg=0.0,
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
			jcn(t) 
			[input=true]
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
			   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn, 
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

struct HHNeuronFSI <: AbstractInhNeuron
    system
    namespace

	function HHNeuronFSI(;
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
			   D(V)~(1/Cₘ)*(-G_Na*m_inf(V)^3*h*(V-E_Na)-G_K*n^2*(V-E_K)-G_L*(V-E_L)-G_D*mD^3*hD*(V-E_K)+I_bg+I_syn+I_gap+I_asc+I_in+σ), 
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
