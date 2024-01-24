function adjmatrixfromdigraph(g::MetaDiGraph)
    myadj = map(Num, adjacency_matrix(g))
    for edge in edges(g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g, edge, :weight)
    end
    return myadj
end

function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

function get_blox(g::MetaDiGraph)
    bs = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if !(b isa AbstractActionSelection)
            push!(bs, b)
        end
    end

    return bs
end

get_sys(g::MetaDiGraph) = get_sys.(get_blox(g))

function connector_from_graph(g::MetaDiGraph)
    bloxs = get_blox(g)
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

function system_from_graph(g::MetaDiGraph; name, t_block=missing)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc; name, t_block)
end

# Additional dispatch if extra parameters are passed for edge definitions
function system_from_graph(g::MetaDiGraph, p::Vector{Num}; name, t_block=missing)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc, p; name, t_block)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector; name, t_block=missing)
    @variables t
    blox_syss = get_sys(g)

    connection_eqs = get_equations_with_state_lhs(bc)
    
    cbs = identity.(get_callbacks(g, bc; t_block))

    return compose(ODESystem(connection_eqs, t, [], params(bc); name, discrete_events = cbs), blox_syss)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector, p::Vector{Num}; name, t_block=missing)
    @variables t
    blox_syss = get_sys(g)

    connection_eqs = get_equations_with_state_lhs(bc)
    
    cbs = identity.(get_callbacks(g, bc; t_block))
    
    return compose(ODESystem(connection_eqs, t, [], vcat(params(bc), p); name, discrete_events = cbs), blox_syss)
end

function system_from_parts(parts::AbstractVector; name)
    @variables t
    return compose(ODESystem(Equation[], t, [], []; name), get_sys.(parts))
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

function create_adjacency_edges!(g::MetaDiGraph, adj_matrix::Matrix{T}) where {T}
    for i = 1:size(adj_matrix, 1)
        for j = 1:size(adj_matrix, 2)
            if !isequal(adj_matrix[i, j], zero(T)) #use isequal because != doesn't work for symbolics
                add_edge!(g, i, j, Dict(:weight => adj_matrix[i, j]))
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