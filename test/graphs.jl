using Neuroblox, Test, SparseArrays, Graphs, MetaGraphs

# testing whether creating a simple graph results in the correct adjacency matrix
# and whether removing a vertex works
g = LinearNeuroGraph(MetaDiGraph())
add_vertex!(g.g)
add_vertex!(g.g)
add_vertex!(g.g)
add_edge!(g.g,1,2,:weight,1.0)
add_edge!(g.g,2,3,:weight,2.0)
add_edge!(g.g,3,1,:weight,3.0)
a = AdjMatrixfromLinearNeuroGraph(g)
rem_vertex!(g.g,2)
b = AdjMatrixfromLinearNeuroGraph(g)
@test a == sparse([3, 1, 2], [1, 2, 3], [3.0, 1.0, 2.0], 3, 3)
@test b == sparse([2], [1], [3.0], 2, 2)
