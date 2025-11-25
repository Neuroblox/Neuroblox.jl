using NeurobloxBasics
using OrdinaryDiffEqDefault, OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner
using StochasticDiffEq
using Test
using Random
using Distributions
using CairoMakie
using ReferenceTests
using StableRNGs

rng = StableRNG(2025)

@testset "LinearNeuralMass" begin
    @named lm1 = LinearNeuralMass()
    @test typeof(lm1) == LinearNeuralMass
end

@testset "LinearNeuralMass + BalloonModel + process noise" begin
    @named lm = LinearNeuralMass()
    @named ou = OUProcess(τ=1, σ=0.1)
    @named bold = BalloonModel()
    g = MetaDiGraph()
    add_blox!.(Ref(g), [lm, ou, bold])
    add_edge!(g, 2, 1, Dict(:weight => 1.0))
    add_edge!(g, 1, 3, Dict(:weight => 0.1))

    @named sys = system_from_graph(g)

    prob = SDEProblem(sys, [], (0.0, 10.0))
    sol = solve(prob, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol.t, sol[:lm₊x])
    @test_reference "plots/linear_neural_mass.png" plt by=psnr_equality(40)
end

@testset "HarmonicOscillator" begin
    @named osc1 = HarmonicOscillator()
    @named osc2 = HarmonicOscillator()

    adj = [0 1; 1 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), [osc1, osc2])
    create_adjacency_edges!(g, adj)

    @named sys = system_from_graph(g, Num[])
    sim_dur = 1e1
    prob = ODEProblem(sys, [], (0.0, sim_dur),[])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    fig, ax, plt = lines(sol.t, sol[:osc1₊x])
    lines!(ax, sol.t, sol[:osc2₊y])
    @test sol.retcode == ReturnCode.Success
    @test_reference "plots/harmonic_oscillator.png" fig by=psnr_equality(40)

    @test param_symbols(HarmonicOscillator) == (:ζ, :k, :h, :ω)
    @test state_symbols(HarmonicOscillator) == (:x, :y)
    @test input_symbols(HarmonicOscillator) == (:jcn,)
    @test output_symbols(HarmonicOscillator) == (:x,)
    @test computed_state_symbols(HarmonicOscillator) == ()
end

@testset "HarmonicOscillator with parameter weights" begin
    @named osc1 = HarmonicOscillator()
    @named osc2 = HarmonicOscillator()

    params = @parameters k=1.0
    adj = [0 k; k 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), [osc1, osc2])
    create_adjacency_edges!(g, adj)

    @named sys = system_from_graph(g, params)
    sim_dur = 1e1
    prob = ODEProblem(sys, [], (0.0, sim_dur), [])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    fig, ax, plt = lines(sol.t, sol[:osc1₊x])
    lines!(ax, sol.t, sol[:osc2₊y])

    @test sol.retcode == ReturnCode.Success
    @test_reference "plots/harmonic_oscillator2.png" fig by=psnr_equality(40)
end

@testset "Jansen-Rit" begin
    τ_factor = 1000
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
    params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

    adj_matrix_lin = [0 0 0 0 0 0 0 0;
                      -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
                      0            -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
                      0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
                      0 0 0 -0.5*C_BG_Th 0 0 0 0;
                      0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
                      0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
                      0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

    g = MetaDiGraph()
    add_blox!.(Ref(g), blox)
    create_adjacency_edges!(g, adj_matrix_lin)

    @named final_system = system_from_graph(g, params)
    final_delays = graph_delays(g)
    sim_dur = 200.0 # Simulate for 2 Seconds
    prob = ODEProblem(final_system,
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
    @named WC1 = WilsonCowan()
    @named WC2 = WilsonCowan()

    adj = [0 1; 1 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), [WC1, WC2])
    create_adjacency_edges!(g, adj)

    @named sys = system_from_graph(g)
    
    sim_dur = 1e2
    prob = ODEProblem(sys, [], (0.0, sim_dur), [])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/wilsoncowan.png" plt by=psnr_equality(40)
end

@testset "Larter-Breakspear" begin
    @named LB1 = LarterBreakspear()
    @named LB2 = LarterBreakspear()

    adj = [0 1; 1 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), [LB1, LB2])
    create_adjacency_edges!(g, adj)

    @named sys = system_from_graph(g)

    sim_dur = 1e2
    prob = ODEProblem(sys, [], (0.0, sim_dur), [])
    sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/larterbreakspear.png" plt by=psnr_equality(40)
end

@testset "Kuramoto Oscillator" begin
    adj = [0 1; 1 0]
    sim_dur = 2e1
    @testset "Non-noisy" begin
        @named K01 = KuramotoOscillator(ω=2.0)
        @named K02 = KuramotoOscillator(ω=5.0)

        g = MetaDiGraph()
        add_blox!.(Ref(g), [K01, K02])
        create_adjacency_edges!(g, adj)

        @named sys = system_from_graph(g)

        prob = ODEProblem(sys, [], (0.0, sim_dur), [])
        sol = solve(prob, Tsit5(), saveat=0.1)
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/kuramoto_non_noisy.png" plt by=psnr_equality(40)
    end

    @testset "Noisy" begin
        @named K01 = KuramotoOscillator(ω=2.0, include_noise=true)
        @named K02 = KuramotoOscillator(ω=5.0, include_noise=true)

        g = MetaDiGraph()
        add_blox!.(Ref(g), [K01, K02])
        create_adjacency_edges!(g, adj)

        @named sys = system_from_graph(g)

        prob = SDEProblem(sys, [], (0.0, sim_dur), [])
        sol = solve(prob, RKMil(), saveat=0.1, seed = rand(rng, 1:100))
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/kuramoto_noisy.png" plt by=psnr_equality(18)
    end
end

@testset "Canonical Micro Circuit network" begin
    # connect multiple canonical micro circuits according to Figure 4 in Bastos et al. 2015
    global_ns = :g # global namespace
    @named r1 = CanonicalMicroCircuit(;namespace=global_ns)
    @named r2 = CanonicalMicroCircuit(;namespace=global_ns)

    g = MetaDiGraph()
    add_blox!.(Ref(g), [r1, r2])

    add_edge!(g, 1, 2, :weightmatrix, [0 1 0 0; # superficial pyramidal to spiny stellate
                                       0 0 0 0;
                                       0 0 0 0;
                                       0 1 0 0]) # superficial pyramidal to deep pyramidal
    # define connections from column (source) to row (sink)
    add_edge!(g, 2, 1, :weightmatrix, [0 0 0  0; 
                                       0 0 0 -1;
                                       0 0 0 -1;
                                       0 0 0  0])
    sys = system_from_graph(g; name=global_ns)

    prob = ODEProblem(sys, [], (0, 10))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/cmc.png" plt by=psnr_equality(40)
end

@testset "OUProcess " begin
    @named ou1 = OUProcess()
    sys = [ou1.system]
    eqs = [sys[1].jcn ~ 0.0]
    @named ou1connected = compose(System(eqs, t; name=:connected),sys)
    ousimpl = structural_simplify(ou1connected)
    prob_ou = SDEProblem(ousimpl,[],(0.0,10.0))
    sol = solve(prob_ou, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[1,:]) > 0.0 # there should be variance

    plt = lines(sol)
    @test_reference "plots/ou_process.png" plt by=psnr_equality(40)
end

@testset "OUProcess & Jansen-Rit network" begin
    @named ou = OUProcess(σ=1.0)
    @named jr = JansenRit()
 
    global_ns = :g # global namespace
    g = MetaDiGraph()
    add_blox!.(Ref(g), [ou, jr])
    add_edge!(g, 1, 2, Dict(:weight => 100.0))
    
    sys = system_from_graph(g, name=global_ns)
    
    prob_oujr = SDEProblem(sys,[],(0.0, 20.0))
    sol = solve(prob_oujr, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[2,:]) > 0.0
    # this test does not make sense, it is true also when JR 
    # is not coupled to OUProcess because of initial conditions at 1 and 
    # then decay test by setting weight to 0.0
    
    plt = lines(sol)
    @test_reference "plots/ou_jansenrit.png" plt by=psnr_equality(30)
end

@testset "OUProcess & Janset-Rit network" begin
    @named ou = OUProcess(σ=5.0)
    @named jr = JansenRit()    
    sys = [ou.system, jr.system]
    eqs = [sys[1].jcn ~ 0.0, sys[2].jcn ~ sys[1].x]
    @named ou1connected = compose(System(eqs, t; name=:connected),sys)
    sys = structural_simplify(ou1connected)
    
    prob_oujr = SDEProblem(sys,[],(0.0, 2.0))
    sol = solve(prob_oujr, ISSEM(); seed = rand(rng, 1:100))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[2,:]) > 0.0 # there should be variance

    plt = lines(sol)
    @test_reference "plots/ou_jansenrit2.png" plt by=psnr_equality(30)
end

@testset "IF Neuron Network" begin
    @named if1 = IFNeuron(I_in=2.5)
    @named if2 = IFNeuron(I_in=1.5)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [if1, if2])
    add_edge!(g, 1, 2, Dict(:weight => -0.008, :connection_rule => "basic"))
    add_edge!(g, 2, 1, Dict(:weight => -0.007, :connection_rule => "basic"))
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 100.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/if_neuron.png" plt by=psnr_equality(40)
end

@testset "LIF Neuron Network" begin
    @named lif1 = LIFNeuron(I_in=2.2)
    @named lif2 = LIFNeuron(I_in=2.1)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [lif1, lif2])
    adj = [0 1; 1 0]
    create_adjacency_edges!(g, adj)
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success 

    plt = lines(sol)
    @test_reference "plots/lif_neuron.png" plt by=psnr_equality(40)
end

@testset "QIF Neuron Network" begin
    @named qif1 = QIFNeuron(I_in=2.5)
    @named qif2 = QIFNeuron(I_in=1.0)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [qif1, qif2])
    add_edge!(g, 1, 2, Dict(:weight => -0.5, :connection_rule => "psp"))
    add_edge!(g, 2, 1, Dict(:weight => 1.0, :connection_rule => "psp"))
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/qif_neuron.png" plt by=psnr_equality(40)
end

@testset "Izhikeveich Neuron Network" begin
    @named izh1 = IzhikevichNeuron()
    @named izh2 = IzhikevichNeuron(η=0.14)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [izh1, izh2])
    add_edge!(g, 1, 2, Dict(:weight => -0.5, :connection_rule => "basic"))
    add_edge!(g, 2, 1, Dict(:weight => 1.0, :connection_rule => "basic"))
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/izh_neuron.png" plt by=psnr_equality(40)
end

@testset "Single Block" begin
    @named solo = JansenRit()
    g = MetaDiGraph()
    add_blox!(g, solo)
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/jansenrit_solo.png" plt by=psnr_equality(40)
end

@testset "LIFExci - LIFInh network" begin
    global_ns = :g # global namespace
    @named n1 = LIFExciNeuron(; namespace = global_ns)
    @named n2 = LIFExciNeuron(; namespace = global_ns)
    @named n3 = LIFInhNeuron(; namespace = global_ns)

    neurons = [n1, n2, n3]
    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    for i in eachindex(neurons)
        for j in eachindex(neurons)
            add_edge!(g, i, j, Dict(:weight => 1))
        end
    end

    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_lifinh.png" plt by=psnr_equality(40)
end

@testset "LIFExciCircuit" begin
    @named n = LIFExciCircuit(; N_neurons = 10, weight=1)

    sys_simpl = structural_simplify(n.system)
    prob = ODEProblem(sys_simpl, [], (0, 200.0))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success 

    plt = meanfield(n, sol)
    @test_reference "plots/lifexci.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain - LIFExci network" begin
    global_ns = :g # global namespace

    tspan = (0, 200) # ms
    spike_rate = 10* 1e-3 # spikes / ms

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, rng = rng)
    @named n1 = LIFExciNeuron(; namespace = global_ns)

    neurons = [s, n1]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain{<:Distribution} - LIFExci network" begin
    global_ns = :g # global namespace

    tspan = (0, 200) # ms
    spike_rate = (distribution=Normal(3, 0.1), dt=10)

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, rng = rng)
    @named n1 = LIFExciNeuron(; namespace = global_ns)

    neurons = [s, n1]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson2.png" plt by=psnr_equality(40)
end

@testset "PoissonSpikeTrain - LIFExciCircuit" begin    
    global_ns = :g # global namespace

    tspan = (0, 1000) # ms
    spike_rate = 10* 1e-3 # spikes / ms

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, rng = rng)
    @named n = LIFExciCircuit(; namespace = global_ns, N_neurons = 10, weight=1)

    neurons = [s, n]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))

    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/lifexci_poisson3.png" plt by=psnr_equality(40)
end

@testset "NGNMM_QIF" begin
    @named exci_PING = NGNMM_QIF(I_ext=10.0, ω=5*2*π/1000, J_internal=8.0, H=1.3, Δ=1.0, τₘ=20.0, A=0.2)
    @named inhi_PING = NGNMM_QIF(I_ext=5.0, ω=5*2*π/1000, J_internal=0.0, H=-5.0, Δ=1.0, τₘ=10.0, A=0.0)

    g = MetaDiGraph()
    add_blox!.(Ref(g), [exci_PING, inhi_PING])
    add_edge!(g, exci_PING => inhi_PING; weight=10.0)
    add_edge!(g, inhi_PING => exci_PING; weight=10.0)

    @named sys = system_from_graph(g)

    sim_dur = 100.0
    prob = SDEProblem(sys, [], (0.0, sim_dur); seed = rand(rng, 1:100))
    sol = solve(prob, RKMil(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/ngnmm_qif.png" plt by=psnr_equality(40)
end

@testset "NGNMM_Izh" begin
    @named popP = NGNMM_Izh(η̄=0.08, κ=0.8, ζ=0.1)
    @named popQ = NGNMM_Izh(η̄=0.08, κ=0.2, wⱼ=0.0095, a=0.077)

    g = MetaDiGraph()
    add_blox!.(Ref(g), [popP, popQ])
    add_edge!(g, popP => popQ; weight=1.0)
    add_edge!(g, popQ => popP; weight=1.0)

    @named sys = system_from_graph(g)

    sim_dur = 200.0
    prob = SDEProblem(sys, [], (0.0, sim_dur))
    sol = solve(prob, RKMil(), saveat=1.0, seed = rand(rng, 1:100))
    @test sol.retcode == ReturnCode.Success

    plt = lines(sol)
    @test_reference "plots/ngnmm_izh.png" plt by=psnr_equality(40)
end

@testset "VdP" begin
    @testset "Non-noisy" begin
        @named vdp = VanDerPol()
        g = MetaDiGraph()
        add_blox!(g, vdp)
        @named sys = system_from_graph(g)
        prob = ODEProblem(sys, [0.0, 0.1], (0.0, 20.0), [])
        sol = solve(prob,Tsit5())
        @test sol.retcode == ReturnCode.Success

        plt = lines( sol)
        @test_reference "plots/vdp_non_noisy.png" plt by=psnr_equality(40)
    end

    @testset "Noisy" begin
        @named vdp = VanDerPol(include_noise=true)
        g = MetaDiGraph()
        add_blox!(g, vdp)
        @named sys = system_from_graph(g)
        prob = SDEProblem(sys, [0.0, 0.1], (0.0, 20.0), [])
        sol = solve(prob, RKMil(); seed = rand(rng, 1:100))
        @test sol.retcode == ReturnCode.Success

        plt = lines(sol)
        @test_reference "plots/vdp_noisy.png" plt by=psnr_equality(40)
    end
end

@testset "ConstantInput - QIFNeuron connection" begin
    @named inp = ConstantInput(; I=4);
    @named blox = QIFNeuron();

    g = MetaDiGraph();
    add_edge!(g, inp => blox, weight = 1);

    @named sys = system_from_graph(g);
    tspan = (0, 100);
    prob = ODEProblem(sys, [], tspan);
    sol = solve(prob, Tsit5());

    @test !all(iszero(detect_spikes(blox, sol)))

    plt = lines(sol)
    @test_reference "plots/qif_constantinput.png" plt by=psnr_equality(40)

    @named inp = ConstantInput(; I=0);

    g = MetaDiGraph();
    add_edge!(g, inp => blox, weight = 1);

    @named sys = system_from_graph(g);
    tspan = (0, 100);
    prob = ODEProblem(sys, [], tspan);
    sol = solve(prob, Tsit5());

    @test all(iszero(detect_spikes(blox, sol)))

    plt = lines(sol)
    @test_reference "plots/qif_constantinput2.png" plt by=psnr_equality(40)
end
