@variables t
D = Differential(t)

function qif_neuron_reduced(;name,C=1.0,E_syn=0, G_syn=1,ω=0,τ=10)
    
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