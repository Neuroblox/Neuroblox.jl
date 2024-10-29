using Neuroblox
using Neuroblox: get_adjacency
using Graphs
using MetaGraphs
using Test
using SparseArrays

@testset "AdjacencyMatrix [HH Neurons]" begin
    @named n1 = HHNeuronExciBlox()
    @named n2 = HHNeuronExciBlox()
    @named n3 = HHNeuronInhibBlox()

    g = MetaDiGraph()
    add_edge!(g, n1 => n2 , weight = 1)
    add_edge!(g, n1 => n3 , weight = 1)
    add_edge!(g, n3 => n2 , weight = 1)
    add_edge!(g, n2 => n2 , weight = 1)

    adj = get_adjacency(g) 

    A = [0 1 1 ; 0 1 0; 0 1 0]

    @test all(A .== adj.matrix)
    @test all([:n1, :n2, :n3] .== adj.names)
end

@testset "AdjacencyMatrix [CorticalBlox]" begin
    global_ns = :g

    A = Matrix{Matrix{Bool}}(undef, 2, 2)
    A[2,1] = [0 1 ; 1 1]
    A[1,2] = [0 1 ; 1 1]

    @named cb1 = CorticalBlox(namespace = global_ns, N_wta=2, N_exci=2, connection_matrices=A, weight=1)

    adj = get_adjacency(cb1) 

    adj_wta_11 = [0 1 1; 1 0 0; 1 0 0]
    adj_wta_12 = [[0 0 0]; hcat([0, 0], A[1,2])]
    adj_wta_21 = [[0 0 0]; hcat([0, 0], A[2,1])]

    A_wta = [adj_wta_11 adj_wta_12 ; adj_wta_21 adj_wta_11]

    A = [
        hcat(A_wta, [0, 0, 0, 0, 0, 0]);
        [0 1 1 0 1 1 0]
    ]

    @test all(A .== adj.matrix)

    nms = [
        :cb1₊wta1₊inh,
        :cb1₊wta1₊exci1,
        :cb1₊wta1₊exci2,
        :cb1₊wta2₊inh,
        :cb1₊wta2₊exci1,
        :cb1₊wta2₊exci2,
        :cb1₊ff_inh
    ]

    @test all(nms .== adj.names)
end

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
