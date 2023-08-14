# creates a winner-take-all local circuit found in neocortex
# typically 5 pyramidal (excitatory) neurons send synapses to a single interneuron (inhibitory)
# and recieve feedback inhibition from that interneuron
@parameters t
D = Differential(t)

mutable struct WinnerTakeAllBlox{N, P, S, C} <: AbstractComponent
    namespace::N
        
        @variables in(t)[1:BlockSize], out(t)[1:BlockSize]

        @named n_inh = HHNeuronInhibBlox(E_syn=E_syn_inhib,G_syn=G_syn_inhib,τ=τ_inhib) 
        n_exci = []
        for ii = 1:BlockSize
            nn = HHNeuronExciBlox(name=Symbol("n_exci$ii"),E_syn=E_syn_exci,G_syn=G_syn_exci,τ=τ_exci,I_in=I_in[ii],freq=freq[ii],phase=phase[ii]) 
            push!(n_exci,nn)
        end
        
        g = MetaDiGraph()
        add_blox!(g,n_inh)
        for jj = 1:BlockSize
            add_blox!(g,n_exci[jj])
        end
        for kk = 1:BlockSize
            add_edge!(g,1,kk+1,:weight,1.0)
            add_edge!(g,kk+1,1,:weight,1.0)
        end

        @named wta_sys = ODEfromGraph(g)
        
        eqs=[]

        for ii= 1:BlockSize
            s=n_exci[ii].odesystem
            push!(eqs,out[ii]~n_exci[ii].connector)
            push!(eqs,in[ii]~s.Isyn)
        end
        
        odesys = extend(ODESystem(eqs, t, name=:connected), wta_sys, name=name)

        new(BlockSize,E_syn_exci,E_syn_inhib,G_syn_exci,G_syn_inhib,I_in,freq,phase,τ_exci,τ_inhib,odesys.out,odesys.in,odesys)
    
    end 

end