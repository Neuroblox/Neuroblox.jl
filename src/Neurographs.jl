using Graphs

struct SimpleNeuroGraph
    g :: SimpleDiGraph
    weights :: Vector{Float64}
    names :: Dict
    blox :: Dict
end

function AdjMatrixfromSimpleNeuroGraph(g::SimpleDiGraph,
                                weights::Vector{Float64})
    myadj = adjacency_matrix(g)
    for (edge,weight) in zip(edges(g),weights)
        myadj[src(edge),dst(edge)] = weight
    end
    myadj
end

function add_edge!(g :: SimpleNeuroGraph, src, dst)
    add_edge!(g.g,src,dst)
end
