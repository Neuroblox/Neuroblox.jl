using
    GraphDynamics,
    Test,
    OrdinaryDiffEqTsit5,
    OrdinaryDiffEqRosenbrock,
    OrdinaryDiffEqVerner,
    Distributions,
    ModelingToolkit,
    Random,
    StochasticDiffEq,
    NeurobloxBasics,
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
        neurons = [IFNeuron(I_in=rand(), name=:lif1)
                IFNeuron(I_in=rand(), name=:lif2)
                QIFNeuron(I_in=rand(), name=:qif1)]
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

function neuron_and_neural_mass_comparison_tests()
    Random.seed!(1234)
    @testset "Comparing GraphDynamics to ModelingToolkit for neuron and neural mass models" begin
        for neurons ∈ ([IFNeuron(I_in=rand(), name=:lif1)
                        IFNeuron(I_in=rand(), name=:lif2)
                        QIFNeuron(I_in=rand(), name=:qif1)],
                       [LIFNeuron(I_in=rand(), name=:lif1)
                        LIFNeuron(I_in=rand(), name=:lif2)],
                       [IzhikevichNeuron(η=rand(), name=:in1)
                        IzhikevichNeuron(η=rand(), name=:in2)],
                       [QIFNeuron(I_in=rand(), name=:qif1)
                        QIFNeuron(I_in=rand(), name=:qif2)
                        WilsonCowan(η=rand(), name=:wc1)
                        WilsonCowan(η=rand(), name=:wc2)],
                       [HarmonicOscillator(name=:ho1)
                        HarmonicOscillator(name=:ho2)
                        JansenRit(name=:jr1)
                        JansenRit(name=:jr2)],
                       [IzhikevichNeuron(η=rand(), name=:in1)
                        IzhikevichNeuron(η=rand(), name=:in2)
                        IFNeuron(I_in=rand(), name=:if1)
                        IFNeuron(I_in=rand(), name=:if2)
                        LIFNeuron(I_in=rand(), name=:lif1)
                        LIFNeuron(I_in=rand(), name=:lif2)
                        QIFNeuron(I_in=rand(), name=:qif1)
                        QIFNeuron(I_in=rand(), name=:qif2)
                        WilsonCowan(η=rand(), name=:wc1)
                        WilsonCowan(η=rand(), name=:wc2)
                        HarmonicOscillator(name=:ho1)
                        HarmonicOscillator(name=:ho2)
                        JansenRit(name=:jr1)
                        JansenRit(name=:jr2)]
                       )
            if length(unknowns(LIFNeuron(;name=:_).system)) > 3
                @warn "excluding LIFNeurons from test"
                filter!(x -> !(x isa LIFNeuron), neurons) # there was a bug in how LIFNeurons were implemented
            end
            if isempty(neurons)
                continue
            end
            @testset "$(join(unique(typeof.(neurons)), ", "))" begin
                g = MetaDiGraph()
                add_blox!.((g,), neurons)
                for i ∈ eachindex(neurons)
                    for j ∈ eachindex(neurons)
                        if i != j
                            if (neurons[i] isa AbstractNeuralMass && neurons[j] isa AbstractNeuron)
                                nothing # Neuroblox doesn't support this currently
                            elseif neurons[i] isa QIFNeuron && neurons[j] isa QIFNeuron
                                add_edge!(g, i, j, Dict(:weight => 2*randn(), :connection_rule => "psp"))
                            elseif neurons[i] isa IFNeuron || neurons[j] isa IFNeuron
                                add_edge!(g, i, j, Dict(:weight => -rand(), :connection_rule => "basic"))
                            else
                                add_edge!(g, i, j, Dict(:weight => 2*randn(), :connection_rule => "basic"))
                            end
                        end 
                    end
                end
                
                tspan = (0.0, 30.0)
                test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-5, alg=Tsit5())
            end
        end
    end
end

function vdp_test()
    @testset "VdP" begin
        Random.seed!(1234)
        @named vdp = VanDerPol()
        g = MetaDiGraph()
        add_blox!(g, vdp)
        test_compare_du_and_sols(ODEProblem, g, (0.0, 1.0); u0map=[:vdp₊x => 0.0, :vdp₊y=>0.1], rtol=1e-10, alg=Vern7())

        @named vdpn = VanDerPol(include_noise=true)
        @named vdpn2 = VanDerPol(include_noise=true)
        g = MetaDiGraph()
        add_blox!(g, vdpn)
        add_blox!(g, vdpn2)
        add_edge!(g, 1, 2, :weight, 1.0)
        
        prob = test_compare_du_and_sols(SDEProblem, g, (0.0, 1.0);
                                        u0map=[:vdpn₊x => 0.0, :vdpn₊y=>1.1], rtol=1e-10, alg=RKMil(), seed=123)
    end
end

function kuramoto_test()
    @testset "Kuramoto Oscillator" begin
        @testset "Non-noisy" begin
            @named K01 = KuramotoOscillator(ω=2.0)
            @named K02 = KuramotoOscillator(ω=5.0)

            adj = [0 1; 1 0]
            g = MetaDiGraph()
            add_blox!.(Ref(g), [K01, K02])
            create_adjacency_edges!(g, adj)

            test_compare_du_and_sols(ODEProblem, g, (0.0, 2.0); rtol=1e-10, alg=AutoVern7(Rodas4()))
        end
        @testset "Noisy" begin
            @named K01 = KuramotoOscillator(ω=2.0, include_noise=true)
            @named K02 = KuramotoOscillator(ω=5.0, include_noise=true)

            adj = [0 1; 1 0]
            g = MetaDiGraph()
            add_blox!.(Ref(g), [K01, K02])
            create_adjacency_edges!(g, adj)

            test_compare_du_and_sols(SDEProblem, g, (0.0, 2.0); rtol=1e-10, alg=RKMil())
        end
    end
end

function lif_exci_inh_tests(;tspan=(0.0, 20.0), rtol=1e-8)
    @testset "LIF Exci / Inhib tests" begin
        ## Describe what the local variables you define are for
        global_ns = :g ## global name for the circuit. All components should be inside this namespace.
        rng = MersenneTwister(1234)

        spike_rate = 2.4 ## spikes / ms

        f = 0.15 ## ratio of selective excitatory to non-selective excitatory neurons
        N_E = 24 ## total number of excitatory neurons
        N_I = Int(ceil(N_E / 4)) ## total number of inhibitory neurons
        N_E_selective = Int(ceil(f * N_E)) ## number of selective excitatory neurons
        N_E_nonselective = N_E - 2 * N_E_selective ## number of non-selective excitatory neurons

        w₊ = 1.7 
        w₋ = 1 - f * (w₊ - 1) / (1 - f)

        ## Use scaling factors for conductance parameters so that our abbreviated model 
        ## can exhibit the same competition behavior between the two selective excitatory populations
        ## as the larger model in Wang 2002 does.
        exci_scaling_factor = 1600 / N_E
        inh_scaling_factor = 400 / N_I

        coherence = 0 # random dot motion coherence [%]
        dt_spike_rate = 50 # update interval for the stimulus spike rate [ms]
        μ_0 = 40e-3 # mean stimulus spike rate [spikes / ms]
        ρ_A = ρ_B = μ_0 / 100
        μ_A = μ_0 + ρ_A * coherence
        μ_B = μ_0 + ρ_B * coherence 
        σ = 4e-3 # standard deviation of stimulus spike rate [spikes / ms]

        spike_rate_A = (distribution=Normal(μ_A, σ), dt=dt_spike_rate) # spike rate distribution for selective population A
        spike_rate_B = (distribution=Normal(μ_B, σ), dt=dt_spike_rate) # spike rate distribution for selective population B

        # Blox definitions
        @named background_input  = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, N_trains=1, rng);
        @named background_input2 = PoissonSpikeTrain(spike_rate + 0.1, tspan; namespace = global_ns, N_trains=1, rng);
        @named stim_A = PoissonSpikeTrain(spike_rate_A, tspan; namespace = global_ns, rng);
        @named stim_B = PoissonSpikeTrain(spike_rate_B, tspan; namespace = global_ns, rng);

        @named n1 = LIFExciNeuron()
        @named n2 = LIFExciNeuron()
        @named n3 = LIFInhNeuron()

        g = MetaDiGraph()

        add_edge!(g, background_input  => n1; weight = 1.0)
        add_edge!(g, background_input2 => n1; weight = 0.0)
        add_edge!(g, stim_A => n1;            weight = 1.0)
        add_edge!(g, stim_B => n1;            weight = 1.0)
        add_edge!(g, n1 => n2;                weight = 1.0)
        add_edge!(g, n2 => n1;                weight = 2.0)
        add_edge!(g, n3 => n1;                weight = 3.0)

        test_compare_du_and_sols(ODEProblem, (deepcopy(g), g), tspan; rtol, alg=Tsit5())
    end
end

function decision_making_test(;tspan=(0.0, 20.0), rtol=1e-5, N_E=24)
    @testset "Decision Making Test" begin 
        ## Describe what the local variables you define are for
        global_ns = :g ## global name for the circuit. All components should be inside this namespace.
        rng = MersenneTwister(1234)
        spike_rate = 2.4 ## spikes / ms

        f = 0.15 ## ratio of selective excitatory to non-selective excitatory neurons
        N_E ## total number of excitatory neurons
        N_I = Int(ceil(N_E / 4)) ## total number of inhibitory neurons
        N_E_selective = Int(ceil(f * N_E)) ## number of selective excitatory neurons
        N_E_nonselective = N_E - 2 * N_E_selective ## number of non-selective excitatory neurons

        w₊ = 1.7 
        w₋ = 1 - f * (w₊ - 1) / (1 - f)

        ## Use scaling factors for conductance parameters so that our abbreviated model 
        ## can exhibit the same competition behavior between the two selective excitatory populations
        ## as the larger model in Wang 2002 does.
        exci_scaling_factor = 1600 / N_E
        inh_scaling_factor = 400 / N_I

        coherence = 0 # random dot motion coherence [%]
        dt_spike_rate = 50 # update interval for the stimulus spike rate [ms]
        μ_0 = 40e-3 # mean stimulus spike rate [spikes / ms]
        ρ_A = ρ_B = μ_0 / 100
        μ_A = μ_0 + ρ_A * coherence
        μ_B = μ_0 + ρ_B * coherence 
        σ = 4e-3 # standard deviation of stimulus spike rate [spikes / ms]

        spike_rate_A = (distribution=Normal(μ_A, σ), dt=dt_spike_rate) # spike rate distribution for selective population A
        spike_rate_B = (distribution=Normal(μ_B, σ), dt=dt_spike_rate) # spike rate distribution for selective population B

        # Blox definitions
        @named background_input = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, N_trains=1, rng);

        @named stim_A = PoissonSpikeTrain(spike_rate_A, tspan; namespace = global_ns, rng);
        @named stim_B = PoissonSpikeTrain(spike_rate_B, tspan; namespace = global_ns, rng);

        @named n_A = LIFExciCircuit(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor);
        @named n_B = LIFExciCircuit(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) ;
        @named n_ns = LIFExciCircuit(; namespace = global_ns, N_neurons = N_E_nonselective, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
        @named n_inh = LIFInhCircuit(; namespace = global_ns, N_neurons = N_I, weight = 1.0, exci_scaling_factor, inh_scaling_factor);

        g = MetaDiGraph()

        add_edge!(g, background_input => n_A; weight = 1)
        add_edge!(g, background_input => n_B; weight = 1)
        add_edge!(g, background_input => n_ns; weight = 1)
        add_edge!(g, background_input => n_inh; weight = 1)

        add_edge!(g, stim_A => n_A; weight = 1)
        add_edge!(g, stim_B => n_B; weight = 1)

        add_edge!(g, n_A => n_B; weight = w₋)
        add_edge!(g, n_A => n_ns; weight = 1)
        add_edge!(g, n_A => n_inh; weight = 1)

        add_edge!(g, n_B => n_A; weight = w₋)
        add_edge!(g, n_B => n_ns; weight = 1)
        add_edge!(g, n_B => n_inh; weight = 1)

        add_edge!(g, n_ns => n_A; weight = w₋)
        add_edge!(g, n_ns => n_B; weight = w₋)
        add_edge!(g, n_ns => n_inh; weight = 1)

        add_edge!(g, n_inh => n_A; weight = 1)
        add_edge!(g, n_inh => n_B; weight = 1)
        add_edge!(g, n_inh => n_ns; weight = 1)

        test_compare_du_and_sols(ODEProblem, (deepcopy(g), g), tspan; rtol, alg=Tsit5())
    end
end

function ping_tests(;tspan=(0.0, 2.0))
    
    # First focus is on producing panels from Figure 1 of the PING network paper.

    # Setup parameters from the supplemental material
    μ_E = 1.5
    σ_E = 0.15
    μ_I = 0.8
    σ_I = 0.08

    # Define the PING network neuron numbers
    NE_driven = 2
    NE_other = 14
    NI_driven = 4
    N_total = NE_driven + NE_other + NI_driven

    # First, create the 20 driven excitatory neurons
    exci_driven = [PINGNeuronExci(name=Symbol("ED$i"), I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NE_driven]
    exci_other  = [PINGNeuronExci(name=Symbol("EO$i")) for i in 1:NE_other]
    inhib       = [PINGNeuronInhib(name=Symbol("ID$i"), I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NI_driven]

    # Create the network
    g = MetaDiGraph()
    add_blox!.(Ref(g), vcat(exci_driven, exci_other, inhib))

    # Extra parameters
    N=N_total
    g_II=0.2
    g_IE=0.6
    g_EI=0.6

    for i = 1:NE_driven+NE_other
        for j = NE_driven+NE_other+1:N_total
            add_edge!(g, i, j, Dict(:weight => g_EI/N))
            add_edge!(g, j, i, Dict(:weight => g_IE/N))
        end
    end

    for i = NE_driven+NE_other+1:N_total
        for j = NE_driven+NE_other+1:N_total
            add_edge!(g, i, j, Dict(:weight => g_II/N))
        end
    end

    test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-7, alg=Tsit5())
end

function auto_tsit5_gdy_test()
    # ensure https://github.com/Neuroblox/GraphDynamics.jl/pull/23 stays fixed
    # This solver involves switching out some types mid solve, which made older
    # versions of GraphDynamics error out.
    @testset "Test AutoTsit5(Rodas4) functionality" begin
        @named neuron = LIFExciNeuron()
        g = MetaDiGraph()
        add_blox!(g, neuron)
        tspan = (0.0, 2500.0)
        @named sys = system_from_graph(g, graphdynamics=true)
        prob = ODEProblem(sys, [], tspan)
        sol = solve(prob, AutoTsit5(Rodas4()))
        @test sol.retcode == ReturnCode.Success

    end
end

function sensitivity_test()
    @testset "Sensitivities" begin
        @testset "Harmonic Oscillator" begin
            @named osc1 = HarmonicOscillator()
            @named osc2 = HarmonicOscillator()
            
            g = GraphSystem()
            add_connection!(g, osc1, osc2; weight=1.0)
            add_connection!(g, osc2, osc1; weight=1.0)
            sim_dur = 1e1
            prob = ODEProblem(g, [], (0.0, sim_dur),[])
            
            test_derivative(1.0) do w
                prob2 = remake(prob, p = [:w_osc1_osc2 => w])
                sol = solve(prob2, Tsit5())
                sol[osc1.x, end]
            end
            test_jacobian([1.0, 2.0, 0.25]; rtol=5e-3) do (w1, w2, ω1)
                prob2 = remake(prob, p = [:w_osc1_osc2 => w1,
                                          :w_osc2_osc1 => w2,
                                          osc1.ω => ω1])
                sol = solve(prob2, Tsit5())
                [sol[osc1.x, end], sol[osc2.x, end]]
            end
        end
        # TODO: More sensitivity tests
    end
end
