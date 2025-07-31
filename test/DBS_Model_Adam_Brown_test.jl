using Neuroblox
using DifferentialEquations
using Test
using Graphs
using MetaGraphs

#test for components used in Adam model for DBS

@testset "Adam_Brown_HH Neuron_network" begin
    nn1 = HHNeuronInhib_MSN_Adam_Blox(name=Symbol("nrn1"))
    nn2 = HHNeuronInhib_FSI_Adam_Blox(name=Symbol("nrn2"), σ=6)
    nn3 = HHNeuronInhib_FSI_Adam_Blox(name=Symbol("nrn3"), σ=6)
    nn4 = HHNeuronExci_STN_Adam_Blox(name=Symbol("nrn4"), σ=8)
    nn5 = HHNeuronInhib_GPe_Adam_Blox(name=Symbol("nrn5"),σ=8)
    assembly = [nn1, nn2, nn3, nn4, nn5]

    # Adjacency matrix : 
    #adj = [0 1 0
    #       0 0 1
    #       0.2 0 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g, 1, 2, Dict(:weight=> 0.1))
    add_edge!(g, 2, 3,  Dict(:weight=> 0.1, :gap => true, :gap_weight=>0.1))
    add_edge!(g, 3, 4, Dict(:weight=> 0.1))
    add_edge!(g, 4, 5, Dict(:weight=> 0.1))
    
    @named neuron_net = system_from_graph(g)
    prob = SDEProblem(structural_simplify(neuron_net), [], (0.0, 2), [])
    sol = solve(prob, ImplicitEM(),saveat = 0.01,reltol=1e-4,abstol=1e-4)
    @test neuron_net isa ODESystem
    @test sol.retcode == ReturnCode.Success
end



@testset "DBS circuit" begin
    global_ns = :g
    @named msn = Striatum_MSN_Adam(namespace=global_ns, N_inhib=2)
    @named fsi = Striatum_FSI_Adam(namespace=global_ns,N_inhib=5)
    @named gpe = GPe_Adam(namespace=global_ns,N_inhib=2)
    @named stn = STN_Adam(namespace=global_ns,N_exci=2)

    assembly = [msn, fsi, gpe, stn]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)

    add_edge!(g, 1, 3, Dict(:weight=> 2.5/100, :density=>0.5))
    add_edge!(g, 2, 1, Dict(:weight=> 0.6/50, :density=>0.5))
    add_edge!(g, 3, 4, Dict(:weight=> 0.3/80, :density=>0.5))
    add_edge!(g, 4, 2, Dict(:weight=> 0.165/40, :density=>0.5))

    @named neuron_net = system_from_graph(g)
    sys = structural_simplify(neuron_net)
    prob = SDEProblem(sys, [], (0.0, 2), [])
    sol = solve(prob)
    @test sys isa ODESystem
    @test sol.retcode == ReturnCode.Success
end