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

function joinmetagraphs(metagraphs::Vector{T}) where T <: Any
    ngraphs = length(metagraphs)
    
    wholegraph = MetaDiGraph()
    nvertex = 0
    for i = 1:ngraphs
        for j in vertices(metagraphs[i].lngraph)
            add_vertex!(wholegraph, props(metagraphs[i].lngraph, j))
        end
        for e in edges(metagraphs[i].lngraph)
            add_edge!(wholegraph, nvertex+src(e), nvertex+dst(e), props(metagraphs[i].lngraph, e))
        end
        nvertex += nv(metagraphs[i].lngraph)
    end
    return wholegraph
end

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
       push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num]))
    end
    return ODESystem(eqs, name=name, systems=sys)
end

function SynapticConnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    syn_eqs = [ 0~sys[1].V - sys[1].V]
    Nrns = length(sys)
    for ii = 1:Nrns
        # for sparse adj matrices it is efficient to only consider presynaptic neurons for computing synaptic input
        presyn = findall(x-> x>0.0, adj_matrix[:,ii])
        wts = adj_matrix[presyn,ii]
        presyn_nrn = sys[presyn]
        postsyn_nrn = sys[ii]
        if length(presyn)>0
            ind = collect(1:length(presyn))
            eqs = [postsyn_nrn.I_syn ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*adj[presyn[p],ii],ind),
                   postsyn_nrn.jcn~postsyn_nrn.I_syn]
            push!(syn_eqs,eqs[1])
            push!(syn_eqs,eqs[2])
        else
            eqs = [postsyn_nrn.I_syn~0,
                  postsyn_nrn.jcn~postsyn_nrn.I_syn]
            push!(syn_eqs,eqs[1])
            push!(syn_eqs,eqs[2])
        end
        
    end
    popfirst!(syn_eqs)
    @named synaptic_eqs = ODESystem(syn_eqs,t)
    synaptic_network = compose(synaptic_eqs, sys;name=name)
    return synaptic_network   
end

function ODEfromGraph(g::MetaDiGraph, jcn; name)
    vert = []
    sys = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        push!(vert, v)
        push!(sys, b.odesystem)
    end
    eqs = []
    for (i, (v, s)) in enumerate(zip(vert, sys))
        if any(occursin.("jcn(t)", string.(states(s)))) # only connect systems with jcn
            if s.jcn isa Symbolics.Arr
                input = []
                for vn in inneighbors(g, v) # vertices that point towards s
                    M = get_prop(g, vn, v, :weightmatrix)
                    connector = get_prop(g, vn, :blox).connector
                    push!(input, M*connector)
                end
                input = sum(input)
                for i = 1:length(s.jcn)
                    push!(eqs, s.jcn[i] ~ input[i])
                end
            else
                input = Num(0)
                for vn in inneighbors(g, v) # vertices that point towards s
                    input += get_prop(g, vn, :blox).connector * get_prop(g, vn, v, :weight)
                end
                push!(eqs, s.jcn ~ input + jcn[i])
            end
        end
    end
    return compose(ODESystem(eqs; name=:connected), sys; name=name)
end

function ODEfromGraph(g::MetaDiGraph ;name)
    eqs = []
    sys = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if isa(b, AbstractBlox) || isa(b, Neuroblox.AbstractComponent)
            s = b.odesystem
            push!(sys, s)
            if any(occursin.("jcn(t)", string.(states(s))))
                if isa(b, AbstractNeuronBlox)
                    input = Num(0)
                    for vn in inneighbors(g, v) # vertices that point towards s
                        bn = get_prop(g, vn, :blox)
                        if !isa(bn, AbstractNeuronBlox) # only neurons can be inputs to neurons
                            continue
                        end
                        input += bn.connector * get_prop(g, vn, v, :weight) * (bn.odesystem.E_syn - s.V)
                    end
                    push!(eqs, s.I_syn ~ input)
                    push!(eqs, s.jcn ~ s.I_syn)
                else
                    if s.jcn isa Symbolics.Arr
                        bi = b.bloxinput # bloxinput only exists if s.jcn isa Symbolics.Arr
                        input = [zeros(Num,length(s.jcn))]
                        for vn in inneighbors(g, v) # vertices that point towards s
                            M = get_prop(g, vn, v, :weightmatrix)
                            connector = get_prop(g, vn, :blox).connector
                            if connector isa Symbolics.Arr
                                connector = collect(connector)
                            end
                            push!(input, vec(M*connector))
                        end
                        input = sum(input)
                        for i = 1:length(s.jcn)
                            push!(eqs, bi[i] ~ input[i])
                        end
                    else
                        input = Num(0)
                        for vn in inneighbors(g, v) # vertices that point towards s
                            connector = get_prop(g,vn,:blox).connector
                            if connector isa Symbolics.Arr
                                input += sum(vec(get_prop(g, vn, v, :weightmatrix)*collect(connector)))
                            else
                                input += connector * get_prop(g, vn, v, :weight)
                            end
                        end
                        if haskey(props(g,v),:jcn)
                            input += get_prop(g,v,:jcn)
                        end
                        push!(eqs, s.jcn ~ input)
                    end
                end
            end
        end
    end
    return compose(ODESystem(eqs, t; name=:connected), sys; name=name)
end

function get_blox(g::MetaDiGraph)
    map(vertices(g)) do v
        get_prop(g, v, :blox)
    end
end

function get_sys(g::MetaDiGraph)
    map(vertices(g)) do v
        b = get_prop(g, v, :blox)
        get_sys(b)
    end
end

# Helper function to get delays from a graph
function graph_delays(g::MetaDiGraph)
    bc = connector_from_graph(g)
    return bc.delays
end

function system_from_graph(g::MetaDiGraph; name)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc; name)
end

# Additional dispatch if extra parameters are passed for edge definitions
function system_from_graph(g::MetaDiGraph, p::Vector{Num}; name)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc, p; name)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector; name)
    @variables t
    blox_syss = get_sys(g)
    return compose(ODESystem(bc.eqs, t, [], params(bc); name, discrete_events = bc.events), blox_syss)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector, p::Vector{Num}; name)
    @variables t
    blox_syss = get_sys(g)
    return compose(ODESystem(bc.eqs, t, [], vcat(params(bc), p); name, discrete_events = bc.events), blox_syss)
end

function system_from_parts(parts::AbstractVector; name)
    @variables t
    return compose(ODESystem(Equation[], t, [], []; name), get_sys.(parts))
end

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

function action_selection_from_graph(g::MetaDiGraph)
    idxs = findall(vertices(g)) do v
        b = get_prop(g, v, :blox)
        b isa AbstractActionSelection
    end

    try
        idx = only(idxs)
    catch
        "Multiple action selection blocks are detected. Only one must be used in an experiment."
    else
        b = get_prop(g, idx, :blox)
        @assert length(inneighbors(g, idx)) == 2 "Two blocks need to connect to the action selection $(nameof(b)) block"

        return b
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

function spikeconnections(;name, sys=sys, psp_amplitude=psp_amplitude, τ=τ, spiketimes=spiketimes)
    psps = psp_amplitude .* exp.(-(t .- spiketimes) ./ τ)
    eqs = []
    for region_num in 1:length(sys)
       push!(eqs, sys[region_num].jcn ~ sum(psps[:, region_num]))
    end
    return ODESystem(eqs, t, name=name, systems=sys)
end

function connectcomplexblox(bloxlist, adjacency_matrices ;name)
    nr = length(bloxlist)
    g = joinmetagraphs(bloxlist)
    row = 0
    for i = 1:nr
        nodes_source = nv(bloxlist[i].lngraph)
        col = 0
        for j = 1:nr
            nodes_sink = nv(bloxlist[j].lngraph)
            if i == j
                col += nodes_sink
                continue
            end
            for idx in CartesianIndices(adjacency_matrices[i, j])
                add_edge!(g, row+idx[1], col+idx[2], :weight, adjacency_matrices[i, j][idx])
            end
            col += nodes_sink
        end
        row += nodes_source
    end
    
    return ODEfromGraph(g, name=name)
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