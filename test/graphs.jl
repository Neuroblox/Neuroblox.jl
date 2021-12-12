using Neuroblox, Test, SparseArrays, Graphs

# testing whether creating a simple graph results in the correct adjacency matrix
# and whether removing a vertex works
g = SimpleNeuroGraph(SimpleDiGraph(),Dict(),Dict(),Dict())
add_vertex!(g,"name1","blox1")
add_vertex!(g,"name2","blox2")
add_vertex!(g,"name3","blox3")
add_edge!(g,1,2,1.0)
add_edge!(g,2,3,2.0)
add_edge!(g,3,1,3.0)
a = AdjMatrixfromSimpleNeuroGraph(g)
rem_vertex!(g,2)
b = AdjMatrixfromSimpleNeuroGraph(g)
@test a == sparse([3, 1, 2], [1, 2, 3], [3.0, 1.0, 2.0], 3, 3)
@test b == sparse([2], [1], [3.0], 2, 2)
