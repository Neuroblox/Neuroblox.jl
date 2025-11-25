using NeurobloxBasics
using Test
using SparseArrays
using Random
using Graphs
using MetaGraphs

@testset "Graph to adjacency matrix" begin
    # testing whether creating a simple graph results in the correct adjacency matrix
    # and whether removing a vertex works
    g = MetaDiGraph()
    add_vertex!(g)
    add_vertex!(g)
    add_vertex!(g)
    add_edge!(g,1,2,:weight,1.0)
    add_edge!(g,2,3,:weight,2.0)
    add_edge!(g,3,1,:weight,3.0)
    a = adjmatrixfromdigraph(g)
    rem_vertex!(g,2)
    b = adjmatrixfromdigraph(g)
    @test a == sparse([3, 1, 2], [1, 2, 3], [3.0, 1.0, 2.0], 3, 3)
    @test b == sparse([2], [1], [3.0], 2, 2)
end

@testset "Add edge blox1 => blox 2" begin
    global_ns = :g
    @named n1 = LIFNeuron(; namespace = global_ns, I_in=2.2)
    @named n2 = LIFNeuron(; namespace = global_ns, I_in=2.1)
    @named n3 = LIFNeuron(; namespace = global_ns)
    g = MetaDiGraph()

    add_edge!(g, n1 => n1; weight = 1)
    @test nv(g) == 1
    add_edge!(g, n1 => n2; weight = 1.2)
    add_edge!(g, n2 => n2; connection_rule = "psp")
    add_edge!(g, n3 => n2)
    add_edge!(g, n1 => n3)

    @test nv(g) == 3
    @test get_prop(g, 1, :blox) == n1
    @test get_prop(g, 2, :blox) == n2
    @test get_prop(g, 3, :blox) == n3
    @test has_edge(g, 1, 2) # n1 => n2
    @test has_edge(g, 2, 2) # n2 => n2
    @test has_edge(g, 1, 3) # n1 => n2

    @test props(g, 1, 2)[:weight] == 1.2
    @test props(g, 2, 2)[:connection_rule] == "psp"
end
