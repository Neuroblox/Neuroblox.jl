@parameters t
D = Differential(t)

#Quadratic Integrate and Fire neurons 
function qif_neuron(;name,C=1.0,E_syn=0, G_syn=1,ω=0,τ=10)
    
    sts = @variables V(t) = -70.0 G(t)=0.0 z(t)=0.0 Isyn(t)=0.0
	ps = @parameters C=C ω=ω I_in=(ω*C/2)^2 Eₘ=0.0 Vᵣₑₛ=-70.0 θ=25 τ₁=τ τ₂=τ E_syn=E_syn G_syn=G_syn
	
	eqs = [
		 D(V) ~ ((V-Eₘ)^2+I_in+Isyn)/C,
		 D(G)~(-1/τ₂)*G + z,
	         D(z)~(-1/τ₁)*z
	      ]
   ev = [V~θ] => [V~Vᵣₑₛ,z~G_syn] 
   return ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
   
end

# Leaky Integrate and Fire neurons
function if_neuron(;name,C=1.0,E_syn=0,G_syn=0.2,I_in=0,freq=0,phase=0,τ=10)
	sts = @variables V(t) = -70.00 G(t)=0.0 z(t)=0.0 spt(t)=0 Cₜ(t) = 0 Isyn(t)=0 
	ps = @parameters C=C I_in = I_in Eₘ = -70.0 Rₘ = 100.0 θ = -50.0 τ₁=0.1 τ₂=τ E_syn=E_syn G_syn=G_syn phase=phase τᵣ=3

	eqs = [
		    D(V) ~ (-(V-Eₘ)/Rₘ + I_in*(sin((t*freq*2*pi/1000)+phase)+1) + Isyn)/(C+Cₜ),
		    D(G)~(-1/τ₂)*G + z,
	        D(z)~(-1/τ₁)*z,
			D(spt)~0,
		    D(Cₜ)~(-1/τᵣ)*Cₜ
		  ]
    ev = [V~θ] => [V~Eₘ,z~G_syn,Cₜ~10]
 
	ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
	
end

# Hodgkin-Huxley Excitatory neurons 
function HH_neuron_excitatory(;name,E_syn=0.0,G_syn=2,I_in=0,freq=0,phase=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0  
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ I_in = I_in freq=freq phase=phase
	
	
    αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
    βₙ(v) = 0.125*exp(-(v+44)/80)

	
    αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
    βₘ(v) = 4*exp(-(v+55)/18)
	 
    αₕ(v) = 0.07*exp(-(v+44)/20)
    βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
    ϕ = 5 
	
    G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in*(sin(t*freq*2*pi/1000+phase)+1)+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end