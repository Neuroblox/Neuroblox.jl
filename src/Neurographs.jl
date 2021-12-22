# define all possible neurographs
abstract type AbstractNeuroGraph end
struct LinearNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end
struct OtherNeuroGraph <: AbstractNeuroGraph; graph::MetaDiGraph end

# method forward does not work
@inline Graphs.add_edge!(g::AbstractNeuroGraph, x...) = add_edge!(g.graph, x...)
@inline Graphs.add_vertex!(g::AbstractNeuroGraph, x...) = add_vertex!(g.graph, x...)
@inline Graphs.rem_vertex!(g::AbstractNeuroGraph, x...) = rem_vertex!(g.graph, x...)

function AdjMatrixfromLinearNeuroGraph(g::LinearNeuroGraph)
    myadj = map(Float64,adjacency_matrix(g.graph))
    for edge in edges(g.graph)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g.graph, edge, :weight)
    end
    myadj
end
