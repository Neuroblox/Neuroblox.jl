struct AdjacencyMatrix
    matrix::SparseMatrixCSC
    names::Vector{Symbol}
end

function AdjacencyMatrix(name)
    return AdjacencyMatrix(spzeros(1,1), [name])
end

function Base.merge(adj1::AdjacencyMatrix, adj2::AdjacencyMatrix)
    return AdjacencyMatrix(
        cat(adj1.matrix, adj2.matrix; dims=(1,2)), 
        vcat(adj1.names, adj2.names)
    )
end

get_adjacency(bc::BloxConnector) = bc.adjacency
get_adjacency(blox::CompositeBlox) = (get_adjacency ∘ get_connector)(blox)
get_adjacency(blox) = AdjacencyMatrix(namespaced_nameof(blox))

function get_adjacency(g::MetaDiGraph)
    bc = connector_from_graph(g)
    return get_adjacency(bc)
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

