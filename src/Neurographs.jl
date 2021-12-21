using Graphs
using MetaGraphs

# define all possible neurographs
abstract type AbstractNeuroGraph end
struct LinearNeuroGraph <: AbstractNeuroGraph; g::MetaDiGraph end
struct OtherNeuroGraph <: AbstractNeuroGraph; g::MetaDiGraph end

# method forward does not work
# add_edges!(ng::AbstractNeuroGraph,args...) = add_edges!(ng.g,args...)
# add_vertex!(ng::AbstractNeuroGraph,args...) = add_vertex!(ng.g,args...)
# gives the following error:
# ERROR: LoadError: error in method definition: function SimpleGraphs.add_vertex! must be explicitly imported to be extended

function AdjMatrixfromLinearNeuroGraph(g::LinearNeuroGraph)
    myadj = map(Float64,adjacency_matrix(g.g))
    for edge in edges(g.g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g.g, edge, :weight)
    end
    myadj
end
