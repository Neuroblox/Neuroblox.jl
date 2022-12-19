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