using
    GraphDynamics,
    Test,
    Distributions,
    Random,
    StochasticDiffEq,
    NeurobloxDBS,
    Random,
    LinearAlgebra,
    CairoMakie,
    StableRNGs,
    ReferenceTests

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using NeurobloxBase: AbstractNeuralMass, AbstractNeuron
using Base.Iterators: map as imap
using GraphDynamics.SymbolicIndexingInterface
using OhMyThreads

rrng = StableRNG(843)

function basic_smoketest()
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
            g = GraphSystem()
            add_node!.((g,), neurons)
            for i ∈ neurons
                for j ∈ neurons
                    weight = 2*randn(rrng)
                    add_connection!(g, i, j; conn = BasicConnection(weight), weight)
                end
            end
            tspan = (0.0, 1.0)

            sol_grp = let prob = ProbType(g, [], tspan, seed = 38)
                sol = solve(prob, alg)
                @test sol.retcode == ReturnCode.Success
                plt = lines(sol)
                @test_reference "plots/gdy_smoketest.png" plt by=psnr_equality(40)
                sol.u[end]
            end
            sol_grp_parallel = let prob = ProbType(g, [], tspan; scheduler=StaticScheduler(), seed = 39)
                sol = solve(prob, alg)
                @test sol.retcode == ReturnCode.Success
                plt = lines(sol)
                @test_reference "plots/gdy_smoketest_parallel.png" plt by=psnr_equality(40)
                sol.u[end]
            end
        end
    end
end

function stochastic_hh_network_tests()
    @testset "Adam_Brown_HH Neuron_network" begin
        g = @graph begin
            @nodes begin
                nn1 = HHNeuronInhib_MSN_Adam(name=Symbol("nrn1"))
                nn2 = HHNeuronInhib_FSI_Adam(name=Symbol("nrn2"), σ=6)
                nn3 = HHNeuronInhib_FSI_Adam(name=Symbol("nrn3"), σ=6)
                nn4 = HHNeuronExci_STN_Adam(name=Symbol("nrn4"), σ=8)
                nn5 = HHNeuronInhib_GPe_Adam(name=Symbol("nrn5"), σ=8)
            end

            @connections begin
                nn1 => nn2, [weight = 0.1]
                nn2 => nn3, [weight = 0.1, gap = true, gap_weight = 0.1]
                nn3 => nn4, [weight = 0.1]
                nn4 => nn5, [weight = 0.1]
            end
        end

        tspan = (0.0, 0.5)
        prob = SDEProblem(g, [], tspan, [], seed = 8)
        sol = solve(prob, RKMil(), saveat = 0.01)
        plt = lines(sol)
        @test_reference "plots/gdy_adam_brown.png" plt by=psnr_equality(40)
    end
    @testset "FSI test 1" begin
        g = @graph begin
            @nodes begin
                n1 = HHNeuronInhib_FSI_Adam(σ=1)
                n2 = HHNeuronInhib_FSI_Adam(σ=2)
            end
            @connections begin
                n1 => n2, [weight = 1.11, gap = false, gap_weight = 1.5]
                n2 => n1, [weight = 1.13, gap = false, gap_weight = 1.0]
            end
        end
        tspan = (0.0, 1.0)
        prob = SDEProblem(g, [], tspan, [], seed = 17)
        sol = solve(prob, RKMil(), saveat = 0.01)
        plt = lines(sol)
        @test_reference "plots/gdy_fsi2.png" plt by=psnr_equality(40)
    end
    @testset "FSI test 2" begin
        g = @graph begin
            @nodes begin
                n1 = HHNeuronInhib_FSI_Adam(σ=1)
                n2 = HHNeuronInhib_FSI_Adam(σ=2)
                n3 = HHNeuronInhib_FSI_Adam(σ=3)
            end
            @connections begin
                n1 => n2, [weight = 100.11, gap = false, gap_weight = 1.5]
                n1 => n3, [weight = 100.12, gap = false]
                n2 => n1, [weight = 100.13, gap = true, gap_weight = 1.0]
                n2 => n3, [weight = 100.0, gap = true, gap_weight = 1.0]
            end
        end
        tspan = (0.0, 1.0)
        prob = SDEProblem(g, [], tspan, [], seed = 190)
        sol = solve(prob, RKMil(), saveat = 0.01)
        plt = lines(sol)
        @test_reference "plots/gdy_fsi3.png" plt by=psnr_equality(40)
    end
end

function dbs_circuit_components()
    @testset "DBS circuit components" begin
        @testset "Striatum_MSN_Adam" begin
            global_ns = :g
            msn = Striatum_MSN_Adam(; name = :msn, namespace=global_ns, N_inhib=10, weight=10.0)
            g = GraphSystem()
            add_node!(g, msn)
            prob = SDEProblem(g, [], (0., .5), [], seed = 19)
            sol = solve(prob, RKMil(), saveat = 0.01)
            plt = meanfield(msn, sol)
            @test_reference "plots/gdy_striatum_msn.png" plt by=psnr_equality(40)
        end
        @testset "Striatum_FSI_Adam" begin
            global_ns = :g
            fsi = Striatum_FSI_Adam(; name = :fsi, namespace=global_ns, N_inhib=10, weight=10.0)
            g = GraphSystem()
            add_node!(g, fsi)
            prob = SDEProblem(g, [], (0., .5), [], seed = 20)
            sol = solve(prob, RKMil(), saveat = 0.01)
            plt = meanfield(fsi, sol)
            @test_reference "plots/gdy_striatum_fsi.png" plt by=psnr_equality(40)
        end
        @testset "GPe_Adam" begin
            global_ns = :g
            gpe = GPe_Adam(; name = :gpe, namespace=global_ns,N_inhib=5)
            g = GraphSystem()
            add_node!(g, gpe)
            prob = SDEProblem(g, [], (0., .5), [], seed = 21)
            sol = solve(prob, RKMil(), saveat = 0.01)
            plt = meanfield(gpe, sol)
            @test_reference "plots/gdy_gpe.png" plt by=psnr_equality(40)
        end
        @testset "STN_Adam" begin
            global_ns = :g
            stn = STN_Adam(; name = :stn, namespace=global_ns,N_exci=2)
            g = GraphSystem()
            add_node!(g, stn)
            prob = SDEProblem(g, [], (0., .5), [], seed = 22)
            sol = solve(prob, RKMil(), saveat = 0.01)
            plt = meanfield(stn, sol)
            @test_reference "plots/gdy_stn.png" plt by=psnr_equality(40)
        end
    end
end

function dbs_circuit()
    @testset "DBS circuit" begin
        global_ns = :g
        make_conn = NeurobloxBase.indegree_constrained_connection_matrix
        num_n(s) = length(s.parts)

        g = @graph begin
            @nodes begin
                msn  = Striatum_MSN_Adam(namespace=global_ns, N_inhib=30)
                fsi  = Striatum_FSI_Adam(namespace=global_ns,N_inhib=40)
                gpe  = GPe_Adam(namespace=global_ns,N_inhib=3)
                stn  = STN_Adam(namespace=global_ns,N_exci=20)
            end
            @connections begin
                msn => gpe, [weight = 2.5/33, connection_matrix = make_conn(0.33, num_n(msn), num_n(gpe))]
                fsi => msn, [weight = 0.6/7.5, connection_matrix = make_conn(0.15, num_n(fsi), num_n(msn))]
                gpe => stn, [weight = 0.3/4, connection_matrix = make_conn(0.05, num_n(gpe), num_n(stn))]
                stn => fsi, [weight = 0.165/4, connection_matrix = make_conn(0.10, num_n(stn), num_n(fsi))]
            end
        end
        
        prob = SDEProblem(g, [], (0., 0.5), [], seed = 489)
        sol = solve(prob, RKMil(), saveat = 0.01)
        fig, ax, plt = meanfield(msn, sol)
        meanfield!(ax, fsi, sol)
        meanfield!(ax, gpe, sol)
        meanfield!(ax, stn, sol)
        @test_reference "plots/gdy_dbscircuit.png" fig by=psnr_equality(40)
    end
end
