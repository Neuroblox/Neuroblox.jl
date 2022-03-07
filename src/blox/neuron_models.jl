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
function IF_neuron(;name,C=1.0,E_syn=0,G_syn=0.2,I_in=0,freq=0,phase=0,τ=10)
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
