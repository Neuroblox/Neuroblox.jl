using NeurobloxBasics
using OrdinaryDiffEqDefault, OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner
using StochasticDiffEq
using Test
using Random
using Distributions
using CairoMakie
using ReferenceTests
using StableRNGs
using ForwardDiff, FiniteDiff

rng = StableRNG(2025)

@testset "LinearNeuralMass" begin
    @named lm1 = LinearNeuralMass()
    @test typeof(lm1) == LinearNeuralMass
end

@testset "LinearNeuralMass + BalloonModel + process noise" begin
    @graph g begin
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ=1, σ=0.1)
            bold = BalloonModel()
        end
        @connections begin
            ou => lm, [weight=2.0]
            lm => bold, [weight=0.1]
        end
    end

    prob = SDEProblem(g, [], (0.0, 10.0))
    sol = solve(prob, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol.t, sol[:g₊lm₊x])
    @test_reference "plots/linear_neural_mass.png" plt by=psnr_equality(25)
end

@testset "HarmonicOscillator" begin
    @graph g begin
        @nodes begin
            osc1 = HarmonicOscillator()
            osc2 = HarmonicOscillator()
        end
        @connections begin
            osc1 => osc2, [weight=1.0]
            osc2 => osc1, [weight=1.0]
        end
    end

    sim_dur = 1e1
    prob = ODEProblem(g, [], (0.0, sim_dur),[])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    fig, ax, plt = lines(sol.t, sol[osc1.x])
    lines!(ax, sol.t, sol[osc2.y])
    @test sol.retcode == ReturnCode.Success
    @test_reference "plots/harmonic_oscillator.png" fig by=psnr_equality(40)

    @test param_symbols(HarmonicOscillator) == (:ω, :ζ, :k, :h)
    @test state_symbols(HarmonicOscillator) == (:x, :y)
    @test input_symbols(HarmonicOscillator) == (:jcn,)
    @test output_symbols(HarmonicOscillator) == (:x,)
    @test computed_property_symbols(HarmonicOscillator) == ()
end

@testset "Jansen-Rit" begin
    τ_factor = 1000
    g = GraphSystem()
    @named str = JansenRit(τ=0.0022*τ_factor, H=20, λ=300, r=0.3)
    @named gpe = JansenRit(τ=0.04*τ_factor, cortical=false) # all default subcortical except τ
    @named stn = JansenRit(τ=0.01*τ_factor, H=20, λ=500, r=0.1)
    @named gpi = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
    @named Th  = JansenRit(τ=0.002*τ_factor, H=10, λ=20, r=5)
    @named EI  = JansenRit(τ=0.01*τ_factor, H=20, λ=5, r=5)
    @named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
    @named II  = JansenRit(τ=2.0*τ_factor, H=60, λ=5, r=5)
    blox = [str, gpe, stn, gpi, Th, EI, PY, II]

    # Store parameters to be passed later on
    C_Cor=60
    C_BG_Th=60
    C_Cor_BG_Th=5
    C_BG_Th_Cor=5

    adj_matrix_lin = [0 0 0 0 0 0 0 0;
                      -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
                      0            -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
                      0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
                      0 0 0 -0.5*C_BG_Th 0 0 0 0;
                      0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
                      0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
                      0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

    for i ∈ axes(adj_matrix_lin, 1)
        for j ∈ axes(adj_matrix_lin, 2)
            if !iszero(adj_matrix_lin[i,j])
                add_connection!(g, blox[i], blox[j]; weight=adj_matrix_lin[i, j])
            end
        end
    end

    sim_dur = 200.0 # Simulate for 2 Seconds
    prob = ODEProblem(g,
        [],
        (0.0, sim_dur))
    alg = Tsit5()
    sol_dde_no_delays = solve(prob, alg, saveat=1)
    @test sol_dde_no_delays.retcode == ReturnCode.Success

    plt = lines(sol_dde_no_delays)
    @test_reference "plots/jansenrit_nodelay.png" plt by=psnr_equality(40)
end

@testset "Jansen-Rit with delay" begin
    @test_broken begin
        τ_factor = 1000
        @named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3, delayed=true)
        @named GPE = JansenRit(τ=0.04*τ_factor, cortical=false, delayed=true) # all default subcortical except τ
        @named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1, delayed=true)
        @named GPI = JansenRit(cortical=false, delayed=true) # default parameters subcortical Jansen Rit blox
        @named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5, delayed=true)
        @named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5, delayed=true)
        @named PY  = JansenRit(cortical=true, delayed=true) # default parameters cortical Jansen Rit blox
        @named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5, delayed=true)
        blox = [Str, GPE, STN, GPI, Th, EI, PY, II]
        g = MetaDiGraph()
        add_blox!.(Ref(g), blox)
        
        # Now, add the edges with the specified delays. Again, if you prefer, there's a version using adjacency and delay matrices to assign all at once.
        add_edge!(g, 2, 1, Dict(:weight => -0.5*60, :delay => 1))
        add_edge!(g, 2, 2, Dict(:weight => -0.5*60, :delay => 2))
        add_edge!(g, 2, 3, Dict(:weight => 60, :delay => 1))
        add_edge!(g, 3, 2, Dict(:weight => -0.5*60, :delay => 1))
        add_edge!(g, 3, 7, Dict(:weight => 5, :delay => 1))
        add_edge!(g, 4, 2, Dict(:weight => -0.5*60, :delay => 4))
        add_edge!(g, 4, 3, Dict(:weight => 60, :delay => 1))
        add_edge!(g, 5, 4, Dict(:weight => -0.5*60, :delay => 2))
        add_edge!(g, 6, 5, Dict(:weight => 5, :delay => 1))
        add_edge!(g, 6, 7, Dict(:weight => 6*60, :delay => 2))
        add_edge!(g, 7, 6, Dict(:weight => 4.8*60, :delay => 3))
        add_edge!(g, 7, 8, Dict(:weight => -1.5*60, :delay => 1))
        add_edge!(g, 8, 7, Dict(:weight => 1.5*60, :delay => 4))
        add_edge!(g, 8, 8, Dict(:weight => 3.3*60, :delay => 1))
        
        # Now you can run the same code as above, but it will handle the delays automatically.
        @named final_system = system_from_graph(g)
        
        # Collect the graph delays and create a DDEProblem.
        final_delays = graph_delays(g)
        sim_dur = 1000.0 # Simulate for 1 second
        prob = DDEProblem(final_system,
            [],
            (0.0, sim_dur),
            constant_lags = final_delays)
        
        # Select the algorihm. MethodOfSteps is now needed because there are non-zero delays.
        alg = MethodOfSteps(Vern7())
        sol_dde_with_delays = solve(prob, alg, saveat=1)
        @test sol_dde_with_delays.retcode == ReturnCode.Success

        plt = lines(sol_dde_with_delays)
        @test_reference "plots/jansenrit_delay.png" plt by=psnr_equality(40)
    end
end

@testset "Wilson-Cowan" begin
    @graph g begin
        @nodes begin
            WC1 = WilsonCowan()
            WC2 = WilsonCowan()
        end
        @connections begin
            WC1 => WC2, [weight=1.0]
            WC2 => WC1, [weight=1.0]
        end
    end
    sim_dur = 1e2
    prob = ODEProblem(g, [], (0.0, sim_dur), [])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/wilsoncowan.png" plt by=psnr_equality(40)
end

@testset "Larter-Breakspear" begin
    @graph g begin
        @nodes begin
            LB1 = LarterBreakspear()
            LB2 = LarterBreakspear()
        end
        @connections begin
            LB1 => LB2, [weight=1.0]
            LB2 => LB1, [weight=1.0]
        end
    end
    sim_dur = 1e2
    prob = ODEProblem(g, [], (0.0, sim_dur), [])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/larterbreakspear.png" plt by=psnr_equality(40)
end

@testset "Kuramoto Oscillator" begin
    sim_dur = 2e1
    rng = StableRNG(13568)
    @testset "Non-noisy" begin
        @graph g begin
            @nodes begin
                K01 = KuramotoOscillator(ω=2.0)
                K02 = KuramotoOscillator(ω=5.0)
            end
            @connections begin
                K01 => K02, [weight=1.0]
                K02 => K01, [weight=1.0]
            end
        end

        prob = ODEProblem(g, [], (0.0, sim_dur), [])
        sol = solve(prob, Tsit5(), saveat=0.1)
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/kuramoto_non_noisy.png" plt by=psnr_equality(40)
    end

    @testset "Noisy" begin
        @graph g begin
            @nodes begin
                K01 = KuramotoOscillator(ω=2.0, include_noise=true)
                K02 = KuramotoOscillator(ω=5.0, include_noise=true)
            end
            @connections begin
                K01 => K02, [weight=1.0]
                K02 => K01, [weight=1.0]
            end
        end

        prob = SDEProblem(g, [], (0.0, sim_dur), [])
        sol = solve(prob, RKMil(), saveat=0.1, seed = rand(rng, 1:100))
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/kuramoto_noisy.png" plt by=psnr_equality(30)
    end
end


@testset "Canonical Micro Circuit network" begin
    # connect multiple canonical micro circuits according to Figure 4 in Bastos et al. 2015
    @graph g begin
        @nodes begin 
            r1 = CanonicalMicroCircuit()
            r2 = CanonicalMicroCircuit()
        end
        @connections begin
            r1 => r2, [weightmatrix = [0 1 0 0; # superficial pyramidal to spiny stellate
                                       0 0 0 0;
                                       0 0 0 0;
                                       0 1 0 0]] # superficial pyramidal to deep pyramidal
            # define connections from column (source) to row (sink)
            r2 => r1, [weightmatrix = [0 0 0  0; 
                                       0 0 0 -1;
                                       0 0 0 -1;
                                       0 0 0  0]]
        end
    end
    prob = ODEProblem(g, [], (0, 10))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/cmc.png" plt by=psnr_equality(40)
end

@testset "OUProcess & Jansen-Rit network" begin
    @graph g begin
        @nodes begin
            ou = OUProcess(σ=1.0)
            jr = JansenRit()
        end
        @connections begin
            ou => jr, [weight=100.0]
        end
    end
    
    prob_oujr = SDEProblem(g,[],(0.0, 20.0))
    sol = solve(prob_oujr, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[2,:]) > 0.0
    # this test does not make sense, it is true also when JR 
    # is not coupled to OUProcess because of initial conditions at 1 and 
    # then decay test by setting weight to 0.0
    
    plt = lines(sol)
    @test_reference "plots/ou_jansenrit.png" plt by=psnr_equality(30)
end


@testset "IF Neuron Network" begin
    @graph g begin
        @nodes begin
            if1 = IFNeuron(I_in=2.5)
            if2 = IFNeuron(I_in=1.5)
        end
        @connections begin
            if1 => if2, [weight=-0.008, connection_rule="basic"]
            if2 => if1, [weight=-0.007, connection_rule="basic"]
        end
    end
    
    prob = ODEProblem(g, [], (0, 100.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/if_neuron.png" plt by=psnr_equality(40)
end

@testset "LIF Neuron Network" begin
    @graph g begin
        @nodes begin
            lif1 = LIFNeuron(I_in=2.2)
            lif2 = LIFNeuron(I_in=2.1)
        end
        @connections begin
            lif1 => lif2, [weight=1.0]
            lif2 => lif1, [weight=1.0]
        end
    end 
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success 
    plt = lines(sol)
    @test_reference "plots/lif_neuron.png" plt by=psnr_equality(40)
end

@testset "QIF Neuron Network" begin
    @graph g begin
        @nodes begin
            qif1 = QIFNeuron(I_in=2.5)
            qif2 = QIFNeuron(I_in=1.0)
        end
        @connections begin
            qif1 => qif2, [weight=-0.5, connection_rule="psp"]
            qif2 => qif1, [weight=1.0, connection_rule="psp"]
        end
    end
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
    plt = lines(sol)
    @test_reference "plots/qif_neuron.png" plt by=psnr_equality(40)
end


@testset "Izhikevich Neuron Network" begin
    @graph g begin
        @nodes begin
            izh1 = IzhikevichNeuron()
            izh2 = IzhikevichNeuron(η=0.14)
        end
        @connections begin
            izh1 => izh2, [weight=-0.5, connection_rule="basic"]
            izh2 => izh1, [weight=1.0, connection_rule="basic"]
        end
    end
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
    plt = lines(sol)
    @test_reference "plots/izh_neuron.png" plt by=psnr_equality(40)
end

@testset "Single Block" begin
    @named solo = JansenRit()
    g = GraphSystem()
    add_node!(g, solo)
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/jansenrit_solo.png" plt by=psnr_equality(40)
end

@testset "LIFExci - LIFInh network" begin
    @graph g begin
        @nodes begin
            n1 = LIFExciNeuron()
            n2 = LIFExciNeuron()
            n3 = LIFInhNeuron()
        end
        @connections begin
            for ni ∈ [n1, n2, n3]
                for nj ∈ [n1, n2, n3]
                    ni => nj, [weight=1.0]
                end
            end
        end
    end
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_lifinh.png" plt by=psnr_equality(40)
end
@testset "LIFExciCircuit" begin
    @named n = LIFExciCircuit(; N_neurons = 10, weight=1)

    prob = ODEProblem(n.graph, [], (0, 200.0))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success 

    plt = meanfield(n, sol)
    @test_reference "plots/lifexci.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain - LIFExci network" begin
    tspan = (0, 200) # ms
    spike_rate = 10 * 1e-3 # spikes / ms
    rng=StableRNG(97531)
    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(spike_rate, tspan; rng = rng)
            n1 = LIFExciNeuron()
        end
        @connections begin
            s => n1, [weight=1.0]
        end
    end
    
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain{<:Distribution} - LIFExci network" begin
    tspan = (0, 200) # ms
    spike_rate = (distribution=Normal(3, 0.1), dt=10)
    rng = StableRNG(086420)
    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(spike_rate, tspan; rng = rng)
            n1 = LIFExciNeuron()
        end
        @connections begin
            s => n1, [weight=1.0]
        end
    end
    
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson2.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain - LIFExciCircuit" begin    
    tspan = (0, 1000) # ms
    spike_rate = 10* 1e-3 # spikes / ms
    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(spike_rate, tspan; rng = StableRNG(34862))
            n = LIFExciCircuit(; N_neurons = 10, weight=1)
        end
        @connections begin
            s => n, [weight=1]
        end
    end
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson3.png" plt by=psnr_equality(30)
end

@testset "NGNMM_QIF" begin
    rng = StableRNG(21864)
    @graph g begin
        @nodes begin
            exci_PING = NGNMM_QIF(I_ext=10.0, ω=5*2*π/1000, J_internal=8.0, H=1.3, Δ=1.0, τₘ=20.0, A=0.2)
            inhi_PING = NGNMM_QIF(I_ext=5.0, ω=5*2*π/1000, J_internal=0.0, H=-5.0, Δ=1.0, τₘ=10.0, A=0.0)
        end
        @connections begin
            exci_PING => inhi_PING, [weight=10.0]
            inhi_PING => exci_PING, [weight=10.0]
        end
    end

    sim_dur = 100.0
    prob = SDEProblem(g, [], (0.0, sim_dur); seed = rand(rng, 1:100))
    sol = solve(prob, RKMil(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/ngnmm_qif.png" plt by=psnr_equality(40)
end

@testset "NGNMM_Izh" begin
    rng = StableRNG(123168923)
    @graph g begin
        @nodes begin
            popP = NGNMM_Izh(η̄=0.08, κ=0.8, ζ=0.1)
            popQ = NGNMM_Izh(η̄=0.08, κ=0.2, wⱼ=0.0095, a=0.077)
        end
        @connections begin
            popP => popQ, [weight=1.0]
            popQ => popP, [weight=1.0]
        end
    end

    sim_dur = 200.0
    prob = SDEProblem(g, [], (0.0, sim_dur))
    sol = solve(prob, RKMil(), saveat=1.0, seed = rand(rng, 1:100))
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/ngnmm_izh.png" plt by=psnr_equality(40)
end

@testset "VdP" begin
    @testset "Non-noisy" begin
        @graph g begin
            @nodes vdp = VanDerPol()
        end
        
        prob = ODEProblem(g, [vdp.x => 0.0, vdp.y => 0.1], (0.0, 20.0), [])
        sol = solve(prob,Tsit5())
        @test sol.retcode == ReturnCode.Success

        plt = lines( sol)
        @test_reference "plots/vdp_non_noisy.png" plt by=psnr_equality(40)
    end
    rng = StableRNG(9842389423)
    @testset "Noisy" begin
        @graph g begin
            @nodes vdp = VanDerPol(include_noise=true)
        end
        
        prob = SDEProblem(g, [vdp.x => 0.0, vdp.y => 0.1], (0.0, 20.0), [])
        sol = solve(prob, RKMil(); seed = rand(rng, 1:100))
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/vdp_noisy.png" plt by=psnr_equality(40)
    end
end

@testset "ConstantInput - QIFNeuron connection" begin
    @graph g begin
        @nodes begin
            inp = ConstantInput(; I=4);
            blox = QIFNeuron();
        end
        @connections inp => blox, [weight = 1]
    end
    tspan = (0, 100);
    prob = ODEProblem(g, [], tspan);
    sol = solve(prob, Tsit5());

    @test !all(iszero(detect_spikes(blox, sol)))

    plt = lines(sol)
    @test_reference "plots/qif_constantinput.png" plt by=psnr_equality(40)

    @graph g2 begin
        @nodes inp2 = ConstantInput(; I=0);
        @connections inp2 => blox, [weight=1.0]
    end
    tspan = (0, 100);
    prob = ODEProblem(g2, [], tspan);
    sol = solve(prob, Tsit5());

    @test all(iszero(detect_spikes(blox, sol)))

    plt = lines(sol)
    @test_reference "plots/qif_constantinput2.png" plt by=psnr_equality(40)
end


@testset "LIF Exci / Inhib tests" begin
    tspan=(0.0, 20.0)
    ## Describe what the local variables you define are for
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
    @graph g begin
        @nodes begin
            background_input  = PoissonSpikeTrain(spike_rate, tspan; N_trains=1, rng);
            background_input2 = PoissonSpikeTrain(spike_rate + 0.1, tspan; N_trains=1, rng);
            stim_A = PoissonSpikeTrain(spike_rate_A, tspan; rng);
            stim_B = PoissonSpikeTrain(spike_rate_B, tspan; rng);

            n1 = LIFExciNeuron()
            n2 = LIFExciNeuron()
            n3 = LIFInhNeuron()
        end
        @connections begin
            background_input  => n1, [weight = 1.0]
            background_input2 => n1, [weight = 0.0]
            stim_A => n1,            [weight = 1.0]
            stim_B => n1,            [weight = 1.0]
            n1 => n2,                [weight = 1.0]
            n2 => n1,                [weight = 2.0]
            n3 => n1,                [weight = 3.0]
        end
    end
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    plt = lines(sol)
    @test_reference "plots/gdy_lif_exci_inhi.png" plt by=psnr_equality(40)
end


@testset "Decision Making Test" begin
    tspan=(0.0, 20.0)
    N_E=24
    ## Describe what the local variables you define are for
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


    @graph g begin
        # Blox definitions
        @nodes begin
            background_input = PoissonSpikeTrain(spike_rate, tspan; N_trains=1, rng);

            stim_A = PoissonSpikeTrain(spike_rate_A, tspan; rng);
            stim_B = PoissonSpikeTrain(spike_rate_B, tspan; rng);

            n_A = LIFExciCircuit(; N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor);
            n_B = LIFExciCircuit(; N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) ;
            n_ns = LIFExciCircuit(; N_neurons = N_E_nonselective, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
            n_inh = LIFInhCircuit(; N_neurons = N_I, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
        end
        @connections begin
            background_input => n_A, [weight = 1]
            background_input => n_B, [weight = 1]
            background_input => n_ns, [weight = 1]
            background_input => n_inh, [weight = 1]

            stim_A => n_A, [weight = 1]
            stim_B => n_B, [weight = 1]

            n_A => n_B, [weight = w₋]
            n_A => n_ns, [weight = 1]
            n_A => n_inh, [weight = 1]

            n_B => n_A, [weight = w₋]
            n_B => n_ns, [weight = 1]
            n_B => n_inh, [weight = 1]

            n_ns => n_A, [weight = w₋]
            n_ns => n_B, [weight = w₋]
            n_ns => n_inh, [weight = 1]

            n_inh => n_A, [weight = 1]
            n_inh => n_B, [weight = 1]
            n_inh => n_ns, [weight = 1]
        end
    end
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    plt = lines(sol)
    @test_reference "plots/gdy_decision.png" plt by=psnr_equality(40)
end

@testset "PING Circuit" begin
    tspan=(0.0, 2.0)
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

    # Extra parameters
    N=N_total
    g_II=0.2
    g_IE=0.6
    g_EI=0.6
    @graph g begin 
        # First, create the 20 driven excitatory neurons
        @nodes begin
            exci_driven = [PINGNeuronExci(I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NE_driven]
            exci_other  = [PINGNeuronExci() for i in 1:NE_other]
            inhib       = [PINGNeuronInhib(I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NI_driven]
        end
        # Create the network
        @connections begin
            for ni = [exci_driven; exci_other]
                for nj = inhib
                    ni => nj, [weight = g_EI/N]
                    nj => ni, [weight = g_IE/N] 
                end
            end
            for ni = inhib
                for nj = inhib
                    ni => nj, [weight = g_II/N]
                end
            end
        end
    end
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, Tsit5())
    plt = lines(sol)
    @test_reference "plots/gdy_ping.png" plt by=psnr_equality(40)
end


# ensure https://github.com/Neuroblox/GraphDynamics.jl/pull/23 stays fixed
# This solver involves switching out some types mid solve, which made older
# versions of GraphDynamics error out.
@testset "Test AutoTsit5(Rodas4) functionality" begin
    @graph g begin
        @nodes neuron = LIFExciNeuron()
    end
    tspan = (0.0, 2500.0)
    prob = ODEProblem(g, [], tspan)
    sol = solve(prob, AutoTsit5(Rodas4()))
    plt = lines(sol)
    @test sol.retcode == ReturnCode.Success
    @test_reference "plots/gdy_autotsit5.png" plt by=psnr_equality(40)
end


