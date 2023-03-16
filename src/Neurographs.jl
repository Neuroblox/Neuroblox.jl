# define all possible neurographs
abstract type AbstractNeuroGraph end
struct LinearNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end
struct OtherNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end

# method forwarding to handle AbstractNeuroGraph
Graphs.add_edge!(g::AbstractNeuroGraph, x...) = add_edge!(g.graph, x...)
Graphs.add_vertex!(g::AbstractNeuroGraph, x...) = add_vertex!(g.graph, x...)
Graphs.rem_vertex!(g::AbstractNeuroGraph, x...) = rem_vertex!(g.graph, x...)

function AdjMatrixfromLinearNeuroGraph(g::LinearNeuroGraph)
    myadj = map(Num, adjacency_matrix(g.graph))
    for edge in edges(g.graph)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g.graph, edge, :weight)
    end
    return myadj
end

function adjmatrixfromdigraph(g::MetaDiGraph)
    myadj = map(Num, adjacency_matrix(g))
    for edge in edges(g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g, edge, :weight)
    end
    return myadj
end

function add_blox!(g::AbstractNeuroGraph,blox)
    add_vertex!(g, :blox, blox)
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

function ODEfromGraph(g::LinearNeuroGraph ;name)
    blox = [get_prop(g.graph, v, :blox) for v in vertices(g.graph)]
    sys = [s.odesystem for s in blox]
    connector = [s.connector for s in blox]
    adj = AdjMatrixfromLinearNeuroGraph(g)
    return LinearConnections(name=name, sys=sys, adj_matrix=adj, connector=connector)
end

function ODEfromGraph(g::MetaDiGraph ;name)
    blox = [get_prop(g, v, :blox) for v in vertices(g)]
    sys = [s.odesystem for s in blox]
    connector = [s.connector for s in blox]
    adj = adjmatrixfromdigraph(g)
    return LinearConnections(name=name, sys=sys, adj_matrix=adj, connector=connector)
end

function ODEfromGraphdirect(g::MetaDiGraph ;name)
    vert = []
    sys = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if isa(b, Neuroblox.Blox) # only use vertices of type Blox for ODESystem
            push!(vert, v)
            push!(sys, b.odesystem)
        end
    end
    eqs = []
    for (v, s) in zip(vert, sys)
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
                    input += get_prop(g,vn,:blox).connector * get_prop(g, vn, v, :weight)
                end
                push!(eqs, s.jcn ~ input)
            end
        end
    end
    return ODESystem(eqs, name=name, systems=sys)
end

function spikeconnections(;name, sys=sys, psp_amplitude=psp_amplitude, τ=τ, spiketimes=spiketimes)
    psps = psp_amplitude .* exp.(-(t .- spiketimes) ./ τ)
    eqs = []
    for region_num in 1:length(sys)
       push!(eqs, sys[region_num].jcn ~ sum(psps[:, region_num]))
    end
    return ODESystem(eqs, name=name, systems=sys)
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