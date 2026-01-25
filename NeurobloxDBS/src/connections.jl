@connection function (conn::BasicConnection)(
    src::Union{HHNeuronInhib_FSI_Adam, HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}, 
    dst::Union{HHNeuronInhib_FSI_Adam, HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam},
    t
   )
    w = conn.weight
    @equations begin
        I_syn = -w * src.G * (dst.V - src.E_syn)
    end
end

# FSI to FSI has a different wiring rule from the others.
@connection function (conn::BasicConnection)(src::HHNeuronInhib_FSI_Adam, dst::HHNeuronInhib_FSI_Adam, t)
    w = conn.weight
    @equations begin
        I_syn = -w * src.Gₛ * (dst.V - src.E_syn)
    end
end

@connection function (conn::HHConnection_GAP)(src::HHNeuronInhib_FSI_Adam, dst::HHNeuronInhib_FSI_Adam, t)
    w = conn.w_gap
    @equations begin
        I_gap = -w * (dst.V - src.V)
    end
end

@connection function (conn::HHConnection_GAP_Reverse)(src::HHNeuronInhib_FSI_Adam, dst::HHNeuronInhib_FSI_Adam, t)
    w = conn.w_gap_rev
    @equations begin
        I_gap = -w * (dst.V - src.V)
    end
end

struct DBSConnection{T} <: ConnectionRule
    weight::T
    stimulus
end
Base.zero(::Type{<:DBSConnection{T}}) where {T} = DBSConnection(zero(T), Returns(zero(T)))

@connection function (conn::DBSConnection)(src::Union{DBS, ProtocolDBS}, dst::AbstractNeuron, t)
    @equations begin
        I_in = conn.weight * conn.stimulus(t)
    end
end

@connection function (conn::DBSConnection)(src::Union{DBS, ProtocolDBS}, dst::AbstractNeuralMass, t)
    @equations begin
        jcn = conn.weight * conn.stimulus(t)
    end
end

@connection function (conn::DBSConnection)(src::Union{DBS, ProtocolDBS}, dst::HHNeuronExci_STN_Adam, t)
    @equations begin
        DBS_in = -dst.V / dst.b + conn.stimulus(t)
    end
end

function GraphDynamics.system_wiring_rule!(g, src::Union{ProtocolDBS, DBS}, dst::Union{AbstractNeuron, AbstractNeuralMass}; weight, kwargs...)
    conn = DBSConnection(weight, src.stimulus)
    add_connection!(g, src, dst; conn, weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, src::Union{ProtocolDBS, DBS}, dst::Union{AbstractComposite}; weight, kwargs...)
    conn = DBSConnection(weight, src.stimulus)
    for comp in get_components(dst)
        add_connection!(g, src, comp; conn, weight, kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g, 
    HH_src::Union{HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}, 
    HH_dst::Union{HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam};
    weight, learning_rule=NoLearningRule(), kwargs...)
    
    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(HH_src.name, "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(HH_dst.name, "spikes_cumulative"))
    
    conn = BasicConnection(weight)
    add_connection!(g, HH_src, HH_dst; conn, weight, learning_rule, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
                                           HH_src::HHNeuronInhib_FSI_Adam, 
                                           HH_dst::HHNeuronInhib_FSI_Adam; weight, gap = false, kwargs...)

    if gap
        gap_weight = get(kwargs, :gap_weight, 0.0)
        # Add a forwards GAP connection from src to dst
        add_connection!(g, HH_src, HH_dst; conn=HHConnection_GAP(gap_weight))
        # Add a reverse GAP connection from the dst to the src so that its I_gap is modified too
        add_connection!(g, HH_dst, HH_src; conn=HHConnection_GAP_Reverse(gap_weight))
    end
    conn = BasicConnection(weight)
    add_connection!(g, HH_src, HH_dst; conn, weight, gap, kwargs...)
end


#----------------------------------------------
# Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam - Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam

function GraphDynamics.system_wiring_rule!(g,
                                           cb_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                                           cb_dst::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam}; kwargs...)
    neurons_src = cb_src.parts 
    neurons_dst = cb_dst.parts

    # TODO: Replace this when the NBBase overhaul happens
    indegree_constrained_connections!(g, neurons_src, neurons_dst,
                                      namespaced_name(inner_namespaceof(cb_src), cb_src.name), namespaced_name(inner_namespaceof(cb_dst), cb_dst.name); kwargs...)
end
