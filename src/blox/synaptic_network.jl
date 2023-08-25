function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix)
    syn_eqs = []
    for ii = 1:length(sys)
        presyn = findall(x-> x>0, adj_matrix[ii,:])
        wts = adj_matrix[ii,presyn]
        presyn_nrn = sys[presyn]
        postsyn_nrn = sys[ii]

        if length(presyn)>0
            ind = collect(1:length(presyn));
            push!(syn_eqs, 0 ~ sum(p -> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*wts[p], ind) - postsyn_nrn.I_syn)
        else
            push!(syn_eqs,0 ~ postsyn_nrn.I_syn);
        end
    end

    @named synaptic_eqs = ODESystem(syn_eqs,t)
    @named synaptic_network = compose(synaptic_eqs, sys)
    return structural_simplify(synaptic_network)
end
