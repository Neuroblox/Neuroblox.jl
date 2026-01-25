struct AdjacencyMatrix
    matrix::SparseMatrixCSC
    names::Vector{Symbol}
end

function AdjacencyMatrix(names::AbstractVector)
    return AdjacencyMatrix(spzeros(1,1), names)
end

AdjacencyMatrix(blox::AbstractComposite) = AdjacencyMatrix(get_graph(blox))

AdjacencyMatrix(blox) = AdjacencyMatrix(namespaced_nameof(blox))

function AdjacencyMatrix(sys::GraphSystem)
    AdjacencyMatrix(GraphSystemParameters(sys))
end
function AdjacencyMatrix(sys::GraphSystemParameters)
    AdjacencyMatrix(sys.connection_matrices, sys.params_partitioned, sys.names_partitioned)
end
function AdjacencyMatrix(prob::ODEProblem)
    p::GraphDynamics.GraphSystemParameters = prob.p
    AdjacencyMatrix(p.connection_matrices, p.params_partitioned, p.names_partitioned)
end
function AdjacencyMatrix(agent::AbstractAgent)
    AdjacencyMatrix(agent.problem)
end

function AdjacencyMatrix(cm::ConnectionMatrices{NConn}, params_partitioned::NTuple{Len, Any}, names_partitioned) where {NConn, Len}
    names, name_index_map = let
        names = Symbol[]
        name_index_map = OrderedDict{Symbol, Int}()
        for i ∈ eachindex(params_partitioned)
            for j ∈ eachindex(params_partitioned[i])
                if !(get_tag(params_partitioned[i][j]) <: AbstractReceptor)
                    push!(names, names_partitioned[i][j])
                end
            end
        end
        perm = sortperm(names)
        names = names[perm]
        names, OrderedDict(name => i for (i, name) ∈ enumerate(names))
    end
    I, J, W = Int[], Int[], Float64[]

    function add_weight!(idx_src, idx_dst, conn)
        weight = get_weight(conn)
        if !isnothing(weight)
            push!(I, idx_src)
            push!(J, idx_dst)
            push!(W, weight)
        end
    end
    
    for i ∈ eachindex(params_partitioned)
        for j ∈ eachindex(params_partitioned[i])
            tag_dst = get_tag(params_partitioned[i][j])
            if !(tag_dst <: AbstractReceptor)
                idx_dst = name_index_map[names_partitioned[i][j]]
                GraphDynamics.foreach_incoming_conn(cm, Val(i), j) do _, k, l, conn
                    tag_src = get_tag(params_partitioned[k][l])
                    if !(tag_src <: AbstractReceptor)
                        idx_src = name_index_map[names_partitioned[k][l]]
                        add_weight!(idx_src, idx_dst, conn)
                    else
                        GraphDynamics.foreach_incoming_conn(cm, Val(k), l) do _, k′, l′, conn′
                            idx_src = name_index_map[names_partitioned[k′][l′]]
                            add_weight!(idx_src, idx_dst, conn) # use conn here, not conn′! it's the connection from synapse to dst that has a real weight
                        end                       
                    end
                end
            end
        end
    end
    AdjacencyMatrix(sparse(I, J, W, length(names), length(names), (l, r) -> error()), names)
end

function Base.merge(adj1::AdjacencyMatrix, adj2::AdjacencyMatrix)
    return AdjacencyMatrix(
        cat(adj1.matrix, adj2.matrix; dims=(1,2)), 
        vcat(adj1.names, adj2.names)
    )
end

"""
    create_adjacency_edges!(g::MetaDiGraph, adj_matrix::Matrix{T}; 
                            kwargs...)

Given an adjacency matrix, populate the graph `g` with the edges whose `weight`s are
stored in  the adjacency matrix. Any additional keyword arguments in `kwargs...` will
be forwarded to `add_connection!`
"""
function create_adjacency_edges!(g::GraphSystem, adj_matrix::AbstractMatrix{T}; kwargs...) where {T}
    v = collect(nodes(g))
    for i ∈ axes(adj_matrix, 1)
        for j ∈ axes(adj_matrix, 2)
            if !iszero(adj_matrix[i, j])
                add_connection!(g, v[i], v[j]; weight = adj_matrix[i, j], kwargs...)
            end
        end
    end
end
