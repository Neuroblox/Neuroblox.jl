function Connector(
    blox_src::Union{HHNeuronInhib_FSI_Adam, HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}, 
    blox_dest::Union{HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
        
    eq = sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::Union{HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}, 
    blox_dest::HHNeuronInhib_FSI_Adam; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
        
    eq = sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::HHNeuronInhib_FSI_Adam,
    blox_dest::HHNeuronInhib_FSI_Adam; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = sys_dest.I_syn ~ -w * sys_src.Gₛ * (sys_dest.V - sys_src.E_syn)

    GAP = get_gap(kwargs, nameof(blox_src), nameof(blox_dest))
    if GAP
        w_gap = generate_gap_weight_param(blox_src, blox_dest; kwargs...)
        eq2 = sys_dest.I_gap ~ -w_gap * (sys_dest.V - sys_src.V)
        eq3 = sys_src.I_gap ~ -w_gap * (sys_src.V - sys_dest.V)

        return Connector(nameof(sys_src), nameof(sys_dest); equation=[eq, eq2, eq3], weight=[w, w_gap])
    else
        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    end
end

function Connector(
    blox_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    blox_dest::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)
    neurons_src = get_inh_neurons(blox_src)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::STN_Adam,
    blox_dest::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_src = get_exci_neurons(blox_src)
    neurons_dest = get_inh_neurons(blox_dest)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    blox_dest::STN_Adam;
    kwargs...
)
    neurons_src = get_inh_neurons(blox_src)
    neurons_dest = get_exci_neurons(blox_dest)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{DBS, ProtocolDBS},
    blox_dest::AbstractComposite;
    kwargs...
)
    components = get_components(blox_dest)
    conn = mapreduce(merge!, components) do comp
        Connector(blox_src, comp; kwargs...)
    end

    return conn
end

function Connector(
    blox_src::Union{DBS, ProtocolDBS},
    blox_dest::AbstractNeuron;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    
    eq = sys_dest.I_in ~ w * sys_src.u
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::Union{DBS, ProtocolDBS},
    blox_dest::AbstractNeuralMass;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    
    eq = sys_dest.jcn ~ w * sys_src.u

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::Union{DBS, ProtocolDBS},
    blox_dest::HHNeuronExci_STN_Adam;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    eq = sys_dest.DBS_in ~ - sys_dest.V/sys_dest.b + sys_src.u
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq)
end
