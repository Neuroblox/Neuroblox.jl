using
    GraphDynamics,
    Test,
    OrdinaryDiffEqTsit5,
    OrdinaryDiffEqVerner,
    Distributions,
    ModelingToolkit,
    Random,
    StochasticDiffEq,
    NeurobloxPharma,
    Random,
    LinearAlgebra

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using NeurobloxBase: AbstractNeuralMass, AbstractNeuron
using NeurobloxBase.GraphDynamicsInterop: t_block_event

using Base.Iterators: map as imap
using GraphDynamics.SymbolicIndexingInterface

using ForwardDiff: ForwardDiff
using FiniteDiff: FiniteDiff
using DiffEqCallbacks: DiffEqCallbacks, PeriodicCallback

using DataFrames: DataFrames, DataFrame
using CSV: CSV

rrng() = Xoshiro(rand(Int))

function basic_smoketest()
    Random.seed!(1234)
    @testset "Basic smoketest" begin
        #let
        # This is just some quick tests to hit some random mechanisms and make sure stuff at least runs before we move
        # on to tests that compare results from GraphDynamics against those from MTK.
        ProbType = ODEProblem 
        alg = Tsit5() 
        neurons = [HHNeuronExci(I_bg=rand(), name=:exci)
                HHNeuronInhib(I_bg=rand(), name=:inh)
                HHNeuronFSI(I_bg=rand(), name=:fsi)]
            
        @testset "$(join(unique(typeof.(neurons)), ", "))" begin
            #let
            nm = NextGenerationEI(name=:ngnmm)
            
            g = MetaDiGraph()
            add_blox!.((g,), neurons)
            add_blox!(g, nm)
            for i ∈ eachindex(neurons)
                add_edge!(g, nm => neurons[i]; weight=1)
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

function basic_hh_network_tests()
    Random.seed!(1234)
    @testset "HH Neuron excitatory & inhibitory network" begin
        nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=3)
        nn2 = HHNeuronExci(name=Symbol("nrn2"), I_bg=2)
        nn3 = HHNeuronInhib(name=Symbol("nrn3"), I_bg=1)
        assembly = [nn1, nn2, nn3]
        # Adjacency matrix : 
        #adj = [0   1 0
        #       0   0 1
        #       0.2 0 0]
        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, :weight, 1.0)
        add_edge!(g, 2, 3, :weight, 1.0)
        add_edge!(g, 3, 1, :weight, 0.2)
        test_compare_du_and_sols(ODEProblem, g, (0.0, 1.0); rtol=1e-10, alg=Vern7())
    end
end

function ngei_test()
    @testset "NextGenerationEI connected to neuron" begin
        global_ns = :g 
        @named LC = NextGenerationEI(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
        @named nn = HHNeuronExci(;namespace=global_ns)
        assembly = [LC, nn]
        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g,1,2, :weight, 44)
        test_compare_du_and_sols(ODEProblem, g, (0.0, 2); rtol=1e-3)
    end
end

function cortical_tests()
    Random.seed!(1234)
    @testset "Cortical blox" begin
        let N_wta = 2, N_exci = 5
            @testset "Single CB" for I_bg ∈ [0, [5 .* rand(N_exci) for _ ∈ 1:N_wta]]
                weight = 1.0
                density = 0.5
                tspan = (0, 1.0)
                namespace = :g
                g = GraphSystem()
                @named cb = Cortical(;N_wta, N_exci, I_bg_ar=0, density, weight, namespace,
                                         rng=Xoshiro(1234))
                add_node!(g, cb)
                test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-1, alg=Tsit5(), parallel=false)
            end
        end
        @testset "CB - CB coupling" begin
            weight = 1.0
            density = 0.5
            tspan = (0, 1.0)
            namespace = :g
            g = GraphSystem()
            @named cb1 = Cortical(;N_wta=2, N_exci=3, I_bg_ar=0, density=0.5, weight, namespace,
                                      rng=Xoshiro(1234))
            @named cb2 = Cortical(;N_wta=3, N_exci=2, I_bg_ar=0, density=0.76, weight, namespace,
                                      rng=Xoshiro(1234))
            add_connection!(g, cb1, cb2; weight=1.0, density=0.25)
            
            test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-1, alg=Tsit5(), parallel=false)
        end
    end
end

function striatum_tests(; sim_len=1.5, t_block=90)
    tspan = (0.0, sim_len)
    Random.seed!(1234)
    @testset "Single Striatum" begin
        let g = MetaDiGraph()
            namespace = :g
            @named s1 = Striatum(;namespace)
            add_blox!(g, s1)
            test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-9, alg=Tsit5())
        end
    end
    @testset "Striatum Coupling" begin
        g = let g = GraphSystem()
            namespace = :g
            
            @named cb = Cortical(;N_wta=2, N_exci=3, I_bg_ar=0, density=0.5, weight=1.0, namespace,
                                     rng=rrng())
            @named s1 = Striatum(;namespace, N_inhib=5,  E_syn_inhib=-71.0, G_syn_inhib=1.1)
            @named s2 = Striatum(;namespace, N_inhib=10, E_syn_inhib=-69.0, G_syn_inhib=1.1)
            
            add_node!(g, s1)
            add_node!(g, s2)
            add_connection!(g, cb, s1; weight = 0.075, density = 0.04, rng=rrng())
            add_connection!(g, cb, s2; weight = 0.075, density = 0.04, rng=rrng())
            add_connection!(g, s1, s2; t_event=181.0, rng=rrng())
            add_connection!(g, s2, s1; t_event=181.0, rng=rrng())
            g
        end

        params_to_compare = [
            :s1₊matrisome₊jcn_,
            :s2₊matrisome₊jcn_,
            :s1₊matrisome₊H_,
            :s2₊matrisome₊H_,
        ]
        test_compare_du_and_sols(ODEProblem, (g), tspan; rtol=1e-9, alg=Tsit5(), params_to_compare, t_block, parallel=true)
    end
end

function wta_tests()
    Random.seed!(1234)
    tspan = (0.0, 1.0)
    N_exci_1 = 5
    I_bg_1 = 5 .* rand(N_exci_1)
    namespace = :g
    @testset "WinnerTakeAll blox" begin
        g1 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            add_blox!(g, wta1)
            g
        end
        g2 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            add_blox!(g, wta1)
            g
        end
        test_compare_du_and_sols(ODEProblem, (g1, g2), tspan; rtol=1e-9, alg=Tsit5())
    end

    N_exci_2 = 5
    I_bg_2 = 5 .* rand(N_exci_2)
    weight = 1.0
    density = 0.25
    
    @testset "WinnerTakeAll network 1" begin
        g1 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            @named wta3 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_blox!(g, wta3)
            add_edge!(g, 1, 2, Dict(:weight => weight, :density => density, :rng => Xoshiro(1234)))
            add_edge!(g, 2, 1, Dict(:weight => weight, :density => density, :rng => Xoshiro(12345)))
            add_edge!(g, 2, 3, Dict(:weight => weight, :density => density, :rng => Xoshiro(1)))
            add_edge!(g, 3, 1, Dict(:weight => weight, :density => density, :rng => Xoshiro(2)))
            g
        end
        g2 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            @named wta3 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_blox!(g, wta3)
            add_edge!(g, 1, 2, Dict(:weight => weight, :density => density, :rng => Xoshiro(1234)))
            add_edge!(g, 2, 1, Dict(:weight => weight, :density => density, :rng => Xoshiro(12345)))
            add_edge!(g, 2, 3, Dict(:weight => weight, :density => density, :rng => Xoshiro(1)))
            add_edge!(g, 3, 1, Dict(:weight => weight, :density => density, :rng => Xoshiro(2)))
            g
        end
        test_compare_du_and_sols(ODEProblem, (g1, g2), tspan; rtol=1e-9, alg=Tsit5())
    end
    @testset "WinnerTakeAll network 2" begin
        density_1_2 = 0.5
        connection_matrix_1_2 = rand(Bernoulli(density_1_2), N_exci_1, N_exci_2)
        g1 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_edge!(g, 1, 2, Dict(:weight => weight, :connection_matrix => connection_matrix_1_2))
            g
        end
        g2 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAll(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAll(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_edge!(g, 1, 2, Dict(:weight => weight, :connection_matrix => connection_matrix_1_2))
            g
        end
        test_compare_du_and_sols(ODEProblem, (g1, g2), tspan; rtol=1e-9, alg=Tsit5())
    end
end


function sta_weight_namemap_test()
    @testset "STA weight namemap test" begin
        g = @graph begin
            @nodes begin
                n1 = HHNeuronExci()
                n2 = HHNeuronExci()
            end
            @connections begin
                n1 => n2, (sta=true, weight=1.5)
            end
        end
        prob = ODEProblem(g, [], (0.0, 1.0), [])
        @test getp(prob, :w_STA_n1_n2)(prob) == 1.5
        setp(prob, :w_STA_n1_n2)(prob, 1.0)
        @test getp(prob, :w_STA_n1_n2)(prob) == 1.0
    end
end
