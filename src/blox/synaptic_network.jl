function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix)
    syn_eqs= [ 0~sys[1].V - sys[1].V]

    for ii = 1:length(sys)
       	
        presyn = findall(x-> x>0, adj_matrix[ii,:])
        wts = adj_matrix[ii,presyn]
		presyn_nrn = sys[presyn]
        postsyn_nrn = sys[ii]
		    
        if length(presyn)>0
            ind = [i for i = 1:length(presyn)];
	        eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*wts[p], ind) - postsyn_nrn.Isyn]
            push!(syn_eqs,eq[1])
			
	else
	    eq = [0~postsyn_nrn.Isyn];
	    push!(syn_eqs,eq[1]);
		 
	end
    end
    popfirst!(syn_eqs)
	
    @named synaptic_eqs = ODESystem(syn_eqs,t)
    
    sys_ode = [sys[ii] for ii = 1:length(sys)]

    @named synaptic_network = compose(synaptic_eqs, sys_ode)
    return structural_simplify(synaptic_network)   

end
