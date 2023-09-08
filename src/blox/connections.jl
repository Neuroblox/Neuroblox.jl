mutable struct BloxConnector
    eqs::Vector{Equation}
    params::Vector{Num}

    BloxConnector() = new(Equation[], Num[])

    function BloxConnector(bloxs)
        eqs = reduce(vcat, input_equations.(bloxs)) 
        params = reduce(vcat, weight_parameters.(bloxs))
        #eqs = namespace_equation.(eqs, nothing, namespace)
        new(eqs, params)
    end
end

function (bc::BloxConnector)(
    HH_out::Union{HHNeuronExciBlox, HHNeuronInhibBlox}, 
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; 
    weight = 1,
    delay = 0
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.params, w)

    eq = sys_in.I_syn ~ w * sys_out.G * (sys_in.V - sys_out.E_syn)
    
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    jc::JansenRitBlox, 
    bloxin; 
    weight = 1,
    delay = 0
)
    # Need t for the delay term
    @variables t

    sys_out = get_namespaced_sys(jc)
    sys_in = get_namespaced_sys(bloxin)

    # Define & accumulate delay parameter
    τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
    τ = only(@parameters $(τ_name)=delay)
    push!(bc.params, τ)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.params, w)

    x = namespace_expr(jc.connector, nothing, nameof(sys_out))
    eq = sys_in.jcn ~ x(t-τ)*w
    
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    wta_out::WinnerTakeAllBlox, 
    wta_in::WinnerTakeAllBlox; 
    weight = 1,
    delay = 0
)
    neurons_in = get_exci_neurons(wta_in)
    neurons_out = get_exci_neurons(wta_out)

    dist = Bernoulli(wta_out.P_connect)

    for neuron_postsyn in neurons_in
        name_postsyn = namespaced_name(namespaceof(neuron_postsyn), nameof(neuron_postsyn))
        for neuron_presyn in neurons_out
            name_presyn = namespaced_name(namespaceof(neuron_presyn), nameof(neuron_presyn))
            # Check names to avoid recurrent connections between the same neuron
            if (name_postsyn != name_presyn) && rand(dist)
                bc(neuron_presyn, neuron_postsyn; weight)
            end
        end
    end
end

function (bc::BloxConnector)(
    cb_out::CorticalBlox,
    cb_in::CorticalBlox;
    weight = 1,
    delay = 0
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)

    out_degree = cb_out.out_degree
    in_degree = Int(ceil(out_degree * length(neurons_in) / length(neurons_out)))
    outgoing_connections = zeros(Int, length(neurons_in))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)))

        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; weight)
        end
        outgoing_connections[idx] .+= 1
    end
end

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs +  eq.rhs

end
