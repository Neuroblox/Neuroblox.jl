using
    GraphDynamics,
    Test,
    Distributions,
    ModelingToolkit,
    Random,
    StochasticDiffEq,
    NeurobloxDBS,
    Random,
    LinearAlgebra

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using NeurobloxBase: AbstractNeuralMass, AbstractNeuron
using Base.Iterators: map as imap
using GraphDynamics.SymbolicIndexingInterface

using ForwardDiff: ForwardDiff
using FiniteDiff: FiniteDiff
using DiffEqCallbacks: DiffEqCallbacks, PeriodicCallback

using NeurobloxBase.GraphDynamicsInterop: t_block_event

rrng() = Xoshiro(rand(Int))

function basic_smoketest()
    Random.seed!(1234)
    @testset "Basic smoketest" begin
        #let
        # This is just some quick tests to hit some random mechanisms and make sure stuff at least runs before we move
        # on to tests that compare results from GraphDynamics against those from MTK.
        ProbType = SDEProblem 
        alg = RKMil()
        neurons = [HHNeuronInhib_GPe_Adam(name=:nrn1, I_bg=3)
                HHNeuronInhib_GPe_Adam(name=:nrn2, I_bg=2)
                HHNeuronExci_STN_Adam(name=:nrn3,  I_bg=1)]
        
        @testset "$(join(unique(typeof.(neurons)), ", "))" begin
            #let
            g = MetaDiGraph()
            add_blox!.((g,), neurons)
            for i ∈ eachindex(neurons)
                for j ∈ eachindex(neurons)
                    add_edge!(g, i, j, Dict(:weight => 2*randn()))
                end
            end
            tspan = (0.0, 1.0)
            @named sys = system_from_graph(g; graphdynamics=true)
            sol_grp = let prob = ProbType(sys, [], tspan)
                sol = solve(prob, alg)
                @test sol.retcode == ReturnCode.Success
                sol.u[end]
            end
            sol_grp_parallel = let prob = ProbType(sys, [], tspan; scheduler=StaticScheduler())
                sol = solve(prob, alg)
                @test sol.retcode == ReturnCode.Success
                sol.u[end]
            end
        end
    end
end

function stochastic_hh_network_tests()
    Random.seed!(1234)
    @testset "Adam_Brown_HH Neuron_network" begin
        nn1 = HHNeuronInhib_MSN_Adam(name=Symbol("nrn1"))
        nn2 = HHNeuronInhib_FSI_Adam(name=Symbol("nrn2"), σ=6)
        nn3 = HHNeuronInhib_FSI_Adam(name=Symbol("nrn3"), σ=6)
        nn4 = HHNeuronExci_STN_Adam(name=Symbol("nrn4"), σ=8)
        nn5 = HHNeuronInhib_GPe_Adam(name=Symbol("nrn5"),σ=8)
        assembly = [nn1, nn2, nn3, nn4, nn5]

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, Dict(:weight=> 0.1))
        add_edge!(g, 2, 3, Dict(:weight=> 0.1, :gap => true, :gap_weight=>0.1))
        add_edge!(g, 3, 4, Dict(:weight=> 0.1))
        add_edge!(g, 4, 5, Dict(:weight=> 0.1))

        tspan = (0.0, 0.5)
        test_compare_du_and_sols(SDEProblem, g, tspan; rtol=1e-10, alg=RKMil())
    end
    @testset "FSI tests" begin
        @named n1 = HHNeuronInhib_FSI_Adam(σ=1)
        @named n2 = HHNeuronInhib_FSI_Adam(σ=2)
        assembly = [n1, n2]

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, Dict(:weight=> 1.11, :gap => false, :gap_weight => 1.5))
        add_edge!(g, 2, 1, Dict(:weight=> 1.13, :gap => false, :gap_weight => 1.0))
        tspan = (0.0, 1.0)
        test_compare_du_and_sols(SDEProblem, g, tspan; rtol=1e-8, alg=RKMil())
    end
    @testset "FSI tests" begin
        @named n1 = HHNeuronInhib_FSI_Adam(σ=1)
        @named n2 = HHNeuronInhib_FSI_Adam(σ=2)
        @named n3 = HHNeuronInhib_FSI_Adam(σ=3)
        assembly = [n1, n2, n3]

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, Dict(:weight=> 100.11, :gap => false, :gap_weight => 1.5))
        add_edge!(g, 1, 3, Dict(:weight=> 100.12, :gap => false))
        add_edge!(g, 2, 1, Dict(:weight=> 100.13, :gap => true, :gap_weight => 1.0))
        add_edge!(g, 2, 3, Dict(:weight=> 100.0,  :gap => true, :gap_weight => 1.0))
        tspan = (0.0, 1.0)
        test_compare_du_and_sols(SDEProblem, g, tspan; rtol=1e-8, alg=RKMil())
    end
end

function dbs_circuit_components()
    @testset "DBS circuit components" begin
        @testset "Striatum_MSN_Adam" begin
            global_ns = :g
            @named msn = Striatum_MSN_Adam(namespace=global_ns, N_inhib=10, weight=10.0)
            g = MetaDiGraph()
            add_blox!(g, msn)
            test_compare_du_and_sols(SDEProblem, g, (0.0, 0.5), rtol=1e-9, alg=RKMil())
        end
        @testset "Striatum_FSI_Adam" begin
            global_ns = :g
            @named msn = Striatum_FSI_Adam(namespace=global_ns, N_inhib=10, weight=10.0)
            g = MetaDiGraph()
            add_blox!(g, msn)
            test_compare_du_and_sols(SDEProblem, g, (0.0, 0.5), rtol=1e-9, alg=RKMil())
        end
        @testset "GPe_Adam" begin
            global_ns = :g
            @named gpe = GPe_Adam(namespace=global_ns,N_inhib=5)
            g = MetaDiGraph()
            add_blox!(g, gpe)
            test_compare_du_and_sols(SDEProblem, g, (0.0, 0.5), rtol=1e-9, alg=RKMil())
        end
        @testset "STN_Adam" begin
            global_ns = :g
            @named stn = STN_Adam(namespace=global_ns,N_exci=2)
            g = MetaDiGraph()
            add_blox!(g, stn)
            test_compare_du_and_sols(SDEProblem, g, (0.0, 0.5), rtol=1e-9, alg=RKMil())
        end
    end
end

function dbs_circuit()
    @testset "DBS circuit" begin
        global_ns = :g
        @named msn  = Striatum_MSN_Adam(namespace=global_ns, N_inhib=30)
        @named fsi  = Striatum_FSI_Adam(namespace=global_ns,N_inhib=40)
        @named gpe  = GPe_Adam(namespace=global_ns,N_inhib=3)
        @named stn  = STN_Adam(namespace=global_ns,N_exci=20)

        assembly = [
            msn,
            fsi,
            gpe,
            stn,
        ]
        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        make_conn = NeurobloxBase.indegree_constrained_connection_matrix
        
        d = Dict(b => i for (i,b) in enumerate(assembly))
        add_edge!(g, 1, 3, Dict(:weight=> 2.5/33,
                                :connection_matrix => make_conn(0.33, length(msn.parts), length(gpe.parts))))
        add_edge!(g, 2, 1, Dict(:weight=> 0.6/7.5,
                                :connection_matrix => make_conn(0.15, length(fsi.parts), length(msn.parts))))
        add_edge!(g, 3, 4, Dict(:weight=> 0.3/4,
                                :connection_matrix => make_conn(0.05, length(gpe.parts), length(stn.parts))))
        add_edge!(g, 4, 2, Dict(:weight=> 0.165/4,
                                :connection_matrix => make_conn(0.10, length(stn.parts), length(fsi.parts))))

        test_compare_du_and_sols(SDEProblem, g, (0.0, 0.5), rtol=1e-5, alg=RKMil(),
                                              sol_comparison_broken=true)
    end
end
