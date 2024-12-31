struct AdjacencyMatrix
    matrix::SparseMatrixCSC
    names::Vector{Symbol}
end

function AdjacencyMatrix(names::AbstractVector)
    return AdjacencyMatrix(spzeros(1,1), names)
end

function AdjacencyMatrix(C::Connector)
    w = weights(C)
    srcs = sources(C)
    dests = destinations(C)
    names = unique(vcat(srcs, dests))
   
    sort!(names)

    ADJ = AdjacencyMatrix(spzeros(length(names), length(names)), names)
    for i in eachindex(srcs)
        add_adjacency_edge!(ADJ, srcs[i], dests[i], w[i])
    end

    return ADJ
end

AdjacencyMatrix(blox::CompositeBlox) = AdjacencyMatrix(get_connector(blox))

AdjacencyMatrix(blox) = AdjacencyMatrix(namespaced_nameof(blox))

AdjacencyMatrix(g::MetaDiGraph) = AdjacencyMatrix(connector_from_graph(g))

function AdjacencyMatrix(bc::Connector, sys::AbstractODESystem, prob::ODEProblem)
    A = AdjacencyMatrix(bc)
    names = A.names
    mat = A.matrix

    I, J, _ = findnz(mat)

    ps = String.(Symbol.(parameters(sys)))

    w_idxs = map(zip(I,J)) do (src_idx, dst_idx)
        w = join(["w", names[src_idx], names[dst_idx]], "_")
        findfirst(p -> p == w, ps)
    end
    
    W = getp(prob, parameters(sys)[w_idxs])(prob)
    S = sparse(I, J, W, size(mat)...)

    return AdjacencyMatrix(S, names)
end

function AdjacencyMatrix(agent::Agent)
    prob = agent.problem
    sys = get_system(agent)
    bc = get_connector(agent)

    return AdjacencyMatrix(bc, sys, prob)
end

function add_adjacency_edge!(ADJ::AdjacencyMatrix, name_src, name_dest, weight)
    src_idx = findfirst(x -> isequal(name_src, x), ADJ.names)
    dest_idx = findfirst(x -> isequal(name_dest, x), ADJ.names)

    weight_def = ModelingToolkit.getdefault(weight)
    weight_value = substitute(weight_def, map(x -> x => ModelingToolkit.getdefault(x), Symbolics.get_variables(weight_def)))
    
    ADJ.matrix[src_idx, dest_idx] = weight_value
end

function Base.merge(adj1::AdjacencyMatrix, adj2::AdjacencyMatrix)
    return AdjacencyMatrix(
        cat(adj1.matrix, adj2.matrix; dims=(1,2)), 
        vcat(adj1.names, adj2.names)
    )
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

