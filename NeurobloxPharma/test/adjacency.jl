using NeurobloxPharma
using SparseArrays
using Random

@testset "AdjacencyMatrix [HH Neurons]" begin
    @named n1 = HHNeuronExci()
    @named n2 = HHNeuronExci()
    @named n3 = HHNeuronInhib()

    g = MetaDiGraph()
    add_edge!(g, n1 => n2 , weight = 1)
    add_edge!(g, n1 => n3 , weight = 1)
    add_edge!(g, n3 => n2 , weight = 1)
    add_edge!(g, n2 => n2 , weight = 1)

    adj = AdjacencyMatrix(g) 

    A = [0 1 1 ; 0 1 0; 0 1 0]

    @test all(A .== adj.matrix)
    @test all([:n1, :n2, :n3] .== adj.names)
end

@testset "AdjacencyMatrix [Cortical]" begin
    global_ns = :g

    A = Matrix{Matrix{Bool}}(undef, 2, 2)
    A[2,1] = [0 1 ; 1 1]
    A[1,2] = [0 1 ; 1 1]

    @named cb1 = Cortical(namespace = global_ns, N_wta=2, N_exci=2, connection_matrices=A, weight=1)

    adj = AdjacencyMatrix(cb1) 

    adj_wta_11 = [0 1 1; 1 0 0; 1 0 0]
    adj_wta_12 = [[0 0 0]; hcat([0, 0], A[1,2])]
    adj_wta_21 = [[0 0 0]; hcat([0, 0], A[2,1])]

    A_wta = [adj_wta_11 adj_wta_12 ; adj_wta_21 adj_wta_11]

    A = [
        hcat(A_wta, [0, 0, 0, 0, 0, 0]);
        [0 1 1 0 1 1 0]
    ]

    @test sum(A) == nnz(adj.matrix)

    nms = [
        :cb1₊wta1₊inh,
        :cb1₊wta1₊exci1,
        :cb1₊wta1₊exci2,
        :cb1₊wta2₊inh,
        :cb1₊wta2₊exci1,
        :cb1₊wta2₊exci2,
        :cb1₊ff_inh
    ]

    @test all(n -> n in nms, adj.names) && length(nms) == length(adj.names)
end

@testset "AdjacencyMatrix [Agent]" begin
    global_namespace = :g

    @named VAC = Cortical(N_wta=2, N_exci=5,  density=0.1, weight=1; namespace=global_namespace) 
    @named AC = Cortical(N_wta=2, N_exci=5, density=0.2, weight=1; namespace=global_namespace) 

    g = MetaDiGraph()

    add_edge!(g, VAC => AC, weight=3, density=0.1)

    Random.seed!(123)
    A_graph = AdjacencyMatrix(g)

    Random.seed!(123)
    agent = Agent(g; name=global_namespace, t_block = 1);
    A_agent = AdjacencyMatrix(agent)

    @test all(A_graph.matrix .== A_agent.matrix)    
    @test all(A_graph.names .== A_agent.names)
end
