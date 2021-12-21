using Graphs
using MetaGraphs

# define all possible neurographs
abstract type AbstractNeuroGraph end
struct LinearNeuroGraph <: AbstractNeuroGraph; g::MetaDiGraph end
struct MichaelesMentonNeuroGraph <: AbstractNeuroGraph; g::MetaDiGraph end

add_edges!(ng::AbstractNeuroGraph,args...) = add_edges!(ng.g,args...)

function AdjMatrixfromMetaDiGraph(g::LinearNeuroGraph)
    myadj = map(Float64,adjacency_matrix(g))
    for edge in edges(g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g, edge, :weight)
    end
    myadj
end
