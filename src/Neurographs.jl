# define all possible neurographs
abstract type AbstractNeuroGraph end
struct LinearNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end
struct OtherNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end

# method forwarding to handle AbstractNeuroGraph
Graphs.add_edge!(g::AbstractNeuroGraph, x...) = add_edge!(g.graph, x...)
Graphs.add_vertex!(g::AbstractNeuroGraph, x...) = add_vertex!(g.graph, x...)
Graphs.rem_vertex!(g::AbstractNeuroGraph, x...) = rem_vertex!(g.graph, x...)

function AdjMatrixfromLinearNeuroGraph(g::LinearNeuroGraph)
    myadj = map(Num,adjacency_matrix(g.graph))
    for edge in edges(g.graph)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g.graph, edge, :weight)
    end
    myadj
end

function add_blox!(g::AbstractNeuroGraph,blox)
    add_vertex!(g,:blox, blox)
end
