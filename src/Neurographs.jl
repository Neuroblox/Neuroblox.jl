function adjmatrixfromdigraph(g::MetaDiGraph)
    myadj = map(Num, adjacency_matrix(g))
    for edge in edges(g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g, edge, :weight)
    end
    return myadj
end

function find_blox(g::MetaDiGraph, blox)
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        b == blox && return v
    end

    return nothing
end

has_blox(g::MetaDiGraph, blox) = isnothing(find_blox(g, blox)) ? false : true

function add_edge!(g::MetaDiGraph, p::Pair; kwargs...)
    src, dest = p
    
    src_idx = find_blox(g, src)
    dest_idx = find_blox(g, dest)
    
    if isnothing(src_idx)
        add_blox!(g, src)
        src_idx = nv(g)
    end
    
    if isnothing(dest_idx)
        add_blox!(g, dest)
        dest_idx = nv(g)
    end
    
    add_edge!(g, src_idx, dest_idx, Dict(kwargs))
end

function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

function get_bloxs(g::MetaDiGraph)
    bs = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if !(b isa AbstractActionSelection)
            push!(bs, b)
        end
    end

    return bs
end

get_sys(g::MetaDiGraph) = get_sys.(get_bloxs(g))

get_dynamics_bloxs(blox) = [blox]
get_dynamics_bloxs(blox::Union{CompositeBlox, AbstractComponent}) = get_blox_parts(blox)

flatten_graph(g::MetaDiGraph) = mapreduce(get_dynamics_bloxs, vcat, get_bloxs(g))

function connector_from_graph(g::MetaDiGraph)
    bloxs = get_bloxs(g)
    link = BloxConnector(bloxs)

    for v in vertices(g)
        b = get_prop(g, v, :blox)
        for vn in inneighbors(g, v)
            bn = get_prop(g, vn, :blox)
            kwargs = props(g, vn, v)
            link(bn, b; kwargs...)
        end
    end
    return link
end

# Helper function to get delays from a graph
function graph_delays(g::MetaDiGraph)
    bc = connector_from_graph(g)
    return bc.delays
end

generate_discrete_callbacks(blox, ::BloxConnector; t_block = missing) = []

function generate_discrete_callbacks(blox::Union{LIFExciNeuron, LIFInhNeuron}, bc::BloxConnector; t_block = missing)
    spike_affect_states = get_spike_affect_states(bc)
    name_blox = namespaced_nameof(blox)

    states_dest = get(spike_affect_states, name_blox, Num[])

    sys = get_namespaced_sys(blox)
    
    cb = (sys.V >= sys.θ) => (
        LIF_spike_affect!, 
        vcat(sys.V, states_dest), 
        [sys.V_reset, sys.t_refract_duration, sys.t_refract_end, sys.is_refractory], 
        [], 
        nothing
    )

    return cb
end

function generate_discrete_callbacks(blox::HHNeuronExciBlox, ::BloxConnector; t_block = missing)
    if !ismissing(t_block)
        nn = get_namespaced_sys(blox)
        eq = nn.spikes_window ~ 0
        cb_spike_reset = (t_block + sqrt(eps(float(t_block)))) => [eq]
        
        return cb_spike_reset
    else
        return []
    end
end

function generate_discrete_callbacks(bc::BloxConnector; t_block = missing)
    eqs_params = get_equations_with_parameter_lhs(bc)

    if !ismissing(t_block) && !isempty(eqs_params)
        cb_params = (t_block - sqrt(eps(float(t_block)))) => eqs_params
        return vcat(cb_params, bc.discrete_callbacks)
    else
        return bc.discrete_callbacks
    end 
end

function generate_discrete_callbacks(g::MetaDiGraph, bc::BloxConnector; t_block = missing)
    bloxs = flatten_graph(g)

    cbs = mapreduce(vcat, bloxs) do blox
        generate_discrete_callbacks(blox, bc; t_block)
    end
    
    cbs_params = generate_discrete_callbacks(bc; t_block)

    return vcat(cbs, cbs_params)
end

function system_from_graph(g::MetaDiGraph, p::Vector{Num}=Num[]; name, t_block=missing, simplify=true, simplify_kwargs...)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc, p; name, t_block, simplify, simplify_kwargs...)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector, p::Vector{Num}=Num[];
                           name, t_block=missing, simplify=true, simplify_kwargs...)
    blox_syss = get_sys(g)
    connection_eqs = get_equations_with_state_lhs(bc)

    discrete_cbs = identity.(generate_discrete_callbacks(g, bc; t_block))

    sys = compose(System(connection_eqs, t, [], vcat(params(bc), p); name, discrete_events = discrete_cbs), blox_syss)
    if simplify
        structural_simplify(sys; simplify_kwargs...)
    else
        sys
    end
end


function system_from_parts(parts::AbstractVector; name)
    return compose(System(Equation[], t; name), get_sys.(parts))
end

function action_selection_from_graph(g::MetaDiGraph)
    idxs = findall(vertices(g)) do v
        b = get_prop(g, v, :blox)
        b isa AbstractActionSelection
    end

    if isempty(idxs)
        #error("No action selection block was detected in the current model.")
        return nothing
    else
        if length(idxs) > 1
            error("Multiple action selection blocks are detected. Only one must be used in an experiment.")
        else
            idx = only(idxs)
            b = get_prop(g, idx, :blox)
            idx_neighbors = inneighbors(g, idx)
            @assert length(idx_neighbors) == 2 "Two blocks need to connect to the action selection $(nameof(b)) block"

            bns = get_prop.(Ref(g), idx_neighbors, :blox)
            connect_action_selection!(b, bns...)

            return b
        end
    end
end

function learning_rules_from_graph(g::MetaDiGraph)
    d = Dict(Num, AbstractLearningRule)()

    for v in vertices(g)
        b = get_prop(g, v, :blox)
        for vn in inneighbors(g, v)
            if has_prop(g, vn, v, :learning_rule)
                bn = get_prop(g, v, :blox)
                weight = get_prop(g, vn, v, :weight)
                w = generate_weight_param(bn, b; weight)
                d[w] = get_prop(g, vn, v, :learning_rule)
            end
        end
    end

    return d
end

## Create Learning Loop
function create_rl_loop(;name, ROIs, datasets, parameters, c_ext)
    # Create LearningBlox for each Region
    regions = []
    for r in eachindex(ROIs)
        push!(regions, 
            LearningBlox(
                ω=parameters[:ω][r], d=parameters[:d][r], 
                prange=vec(datasets[r][1]), pdata=vec(datasets[r][2]), 
                name=ROIs[r]
            )
        )
    end
    # Connect Regions through an External Connection Weight
    @parameters c_ext=c_ext
    for r in eachindex(ROIs)
        regions[r].adj[size(regions[r].adj, 1), size(regions[r].adj, 2)] = c_ext*regions[1:end .!= r, :][1].sys[3].x
    end
    # Update Adjacency Matrix to Incorporate External Connections
    eqs = []
    for r in eachindex(ROIs)
        for s in eachindex(regions[r].sys) 
            push!(eqs, regions[r].sys[s].jcn ~ sum(regions[r].adj[:, s]))
        end
    end
    # Compose Loop
    sys = []
    for r in eachindex(ROIs)
        sys = vcat(sys, regions[r].sys)
    end
    # Return One ODESystem
    return ODESystem(eqs, systems=sys, name=name)
end

function create_adjacency_edges!(g::MetaDiGraph, adj_matrix::Matrix{T}; connection_rule="basic") where {T}
    for i = 1:size(adj_matrix, 1)
        for j = 1:size(adj_matrix, 2)
            if !isequal(adj_matrix[i, j], zero(T)) #use isequal because != doesn't work for symbolics
                add_edge!(g, i, j, Dict(:weight => adj_matrix[i, j], :connection_rule => connection_rule))
            end
        end
    end
end

function create_adjacency_edges!(g::MetaDiGraph, adj_matrix::Matrix{T}, delay_matrix) where {T}
    for i = 1:size(adj_matrix, 1)
        for j = 1:size(adj_matrix, 2)
            if !isequal(adj_matrix[i, j], zero(T)) #use isequal because != doesn't work for symbolics
                add_edge!(g, i, j, Dict(:weight => adj_matrix[i, j], :delay => delay_matrix[i, j]))
            end
        end
    end
end
