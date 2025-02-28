using Neuroblox
using OrdinaryDiffEq
using StochasticDiffEq
using DelayDiffEq
using DataFrames
using Test
using Distributions
using Statistics
using LinearAlgebra
using Graphs
using MetaGraphs
using Random

@testset "LinearNeuralMass" begin
    @named lm1 = LinearNeuralMass()
    @test typeof(lm1) == LinearNeuralMass
end

@testset "LinearNeuralMass + BalloonModel + process noise" begin
    @named lm = LinearNeuralMass()
    @named ou = OUBlox(τ=1, σ=0.1)
    @named bold = BalloonModel()
    g = MetaDiGraph()
    add_blox!.(Ref(g), [lm, ou, bold])
    add_edge!(g, 2, 1, Dict(:weight => 1.0))
    add_edge!(g, 1, 3, Dict(:weight => 0.1))

    @named sys = system_from_graph(g)

    prob = SDEProblem(sys, [], (0.0, 10.0))
    sol = solve(prob, ISSEM())
    @test sol.retcode == ReturnCode.Success
end


"""
HarmonicOscillator tests
"""

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
    @test sol.retcode == ReturnCode.Success
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
    @test sol.retcode == ReturnCode.Success
end

"""
New Jansen-Rit tests
"""

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
    sim_dur = 2000.0 # Simulate for 2 Seconds
    prob = ODEProblem(final_system,
        [],
        (0.0, sim_dur))
    alg = Vern7()
    sol_dde_no_delays = solve(prob, alg, saveat=1)
    @test sol_dde_no_delays.retcode == ReturnCode.Success
end

@testset "Jansen-Rit with delay" begin
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
        sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "Noisy" begin
        @named K01 = KuramotoOscillator(ω=2.0, include_noise=true)
        @named K02 = KuramotoOscillator(ω=5.0, include_noise=true)

        g = MetaDiGraph()
        add_blox!.(Ref(g), [K01, K02])
        create_adjacency_edges!(g, adj)

        @named sys = system_from_graph(g)

        prob = SDEProblem(sys, [], (0.0, sim_dur), [])
        sol = solve(prob, RKMil(), saveat=0.1)
        @test sol.retcode == ReturnCode.Success
    end
end

@testset "Canonical Micro Circuit network" begin
    # connect multiple canonical micro circuits according to Figure 4 in Bastos et al. 2015
    global_ns = :g # global namespace
    @named r1 = CanonicalMicroCircuitBlox(;namespace=global_ns)
    @named r2 = CanonicalMicroCircuitBlox(;namespace=global_ns)

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
    sum(sol[end, 2:end])
    @test sol.retcode == ReturnCode.Success
end

"""
stochastic.jl test

Test for OUBlox generator.
"""

@testset "OUBlox " begin
    @named ou1 = OUBlox()
    sys = [ou1.system]
    eqs = [sys[1].jcn ~ 0.0]
    @named ou1connected = compose(System(eqs, t; name=:connected),sys)
    ousimpl = structural_simplify(ou1connected)
    prob_ou = SDEProblem(ousimpl,[],(0.0,10.0))
    sol = solve(prob_ou, ISSEM())
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[1,:]) > 0.0 # there should be variance
end

@testset "OUBlox & Janset-Rit network" begin
    @named ou = OUBlox(σ=1.0)
    @named jr = JansenRit()
 
    global_ns = :g # global namespace
    g = MetaDiGraph()
    add_blox!.(Ref(g), [ou, jr])
    add_edge!(g, 1, 2, Dict(:weight => 100.0))
    
    sys = system_from_graph(g, name=global_ns)
    
    prob_oujr = SDEProblem(sys,[],(0.0, 20.0))
    sol = solve(prob_oujr, ISSEM())
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[2,:]) > 0.0
    # this test does not make sense, it is true also when JR 
    # is not coupled to OU because of initial conditions at 1 and 
    # then decay test by setting weight to 0.0
end

@testset "OUBlox & Janset-Rit network" begin
    @named ou = OUBlox(σ=5.0)
    @named jr = JansenRit()    
    sys = [ou.system, jr.system]
    eqs = [sys[1].jcn ~ 0.0, sys[2].jcn ~ sys[1].x]
    @named ou1connected = compose(System(eqs, t; name=:connected),sys)
    sys = structural_simplify(ou1connected)
    
    prob_oujr = SDEProblem(sys,[],(0.0, 2.0))
    sol = solve(prob_oujr, ISSEM())
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test std(sol[2,:]) > 0.0 # there should be variance
end

@testset "HH Neuron excitatory & inhibitory network" begin
    nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=3, freq=4)
    nn2 = HHNeuronExciBlox(name=Symbol("nrn2"), I_bg=2, freq=6)
    nn3 = HHNeuronInhibBlox(name=Symbol("nrn3"), I_bg=2, freq=3)
    assembly = [nn1, nn2, nn3]

    # Adjacency matrix : 
    #adj = [0 1 0
    #       0 0 1
    #       0.2 0 0]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g, 1, 2, :weight, 1)
    add_edge!(g, 2, 3, :weight, 1)
    add_edge!(g, 3, 1, :weight, 0.2)
    
    @named neuron_net = system_from_graph(g)
    prob = ODEProblem(neuron_net, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test neuron_net isa ODESystem
    @test sol.retcode == ReturnCode.Success
end

@testset "NGNMM_theta connected to neuron" begin
    global_ns = :g 
    @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
    @named nn = HHNeuronExciBlox(;namespace=global_ns)
    assembly = [LC, nn]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(neuron_net, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test neuron_net isa ODESystem
    @test sol.retcode == ReturnCode.Success
end

@testset "NGNMM_theta connected to CorticalBlox" begin
    global_ns = :g 
    @named LC = NGNMM_theta(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
    @named cb = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    assembly = [LC, cb]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(neuron_net, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success
end

@testset "WinnerTakeAll" begin
    N_exci = 5
    @named wta= WinnerTakeAllBlox(;I_bg=5.0*rand(N_exci), N_exci)
    sys = wta.system
    wta_simp=structural_simplify(sys)
    prob = ODEProblem(wta_simp,[],(0,10))
    sol = solve(prob, Vern7(), saveat=0.1)

    @test wta_simp isa ODESystem
    @test sol.retcode == ReturnCode.Success 
end

@testset "WinnerTakeAll network" begin
    global_ns = :g # global namespace
    N_exci = 5
    @named wta1 = WinnerTakeAllBlox(;I_bg=5.0, N_exci, namespace=global_ns)
    @named wta2 = WinnerTakeAllBlox(;I_bg=5.0, N_exci, namespace=global_ns)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [wta1, wta2])
    add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.5))
    sys = system_from_graph(g; name=global_ns)

    prob = ODEProblem(sys, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success 
end

@testset "Cortical" begin
    @named cb = CorticalBlox(N_wta=6, N_exci=5, density=0.1, weight=1)
    cb_simpl = structural_simplify(cb.system)
    prob = ODEProblem(cb_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "Striatum" begin
    @named str_scb = Striatum(N_inhib=2)
    str_simpl = structural_simplify(str_scb.system)
    prob = ODEProblem(str_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "GPi" begin
    @named gpi_scb = GPi(N_inhib=2)
    gpi_simpl = structural_simplify(gpi_scb.system)
    prob = ODEProblem(gpi_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "GPe" begin
    @named gpe_scb = GPe(N_inhib=2)
    gpe_simpl = structural_simplify(gpe_scb.system)
    prob = ODEProblem(gpe_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "STN" begin
    @named stn_scb = STN(N_exci=2)
    stn_simpl = structural_simplify(stn_scb.system)
    prob = ODEProblem(stn_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "Thalamus" begin
    @named thal_scb = Thalamus(N_exci=2)
    thal_simpl = structural_simplify(thal_scb.system)
    prob = ODEProblem(thal_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 
end

@testset "Cortical-ImageStimulus network" begin
    global_ns = :g # global namespace
    @named cb = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    fn = joinpath(@__DIR__, "../examples/image_example.csv")
    @named stim = ImageStimulus(fn; namespace=global_ns, t_stimulus=1, t_pause=0.5)
    g = MetaDiGraph()
    add_blox!(g, stim)
    add_blox!(g, cb)
    add_edge!(g, 1, 2, :weight, 1)
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], (0, 2))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success 
end

@testset "Cortical-Cortical network" begin
    global_ns = :g # global namespace
    @named cb1 = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    @named cb2 = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [cb1, cb2])
    add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.1))
    sys = system_from_graph(g; name=global_ns, t_block=90.0)
    prob = ODEProblem(sys, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success 
end

@testset "Cortical & subcortical components network" begin
    global_ns = :g # global namespace
    @named cb1 = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named cb2 = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    @named str1 = Striatum(N_inhib=2, namespace=global_ns)
    @named gpi1 = GPi(N_inhib=2, namespace=global_ns)
    @named thal1 = Thalamus(N_exci=2, namespace=global_ns)

    g = MetaDiGraph()
    add_blox!.(Ref(g), [cb1, cb2, str1, gpi1, thal1])
    add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.1))
    add_edge!(g, 2, 3, Dict(:weight => 1, :density => 0.1))
    add_edge!(g, 3, 4, Dict(:weight => 1, :density => 0.1))
    add_edge!(g, 4, 5, Dict(:weight => 1, :density => 0.1))
    add_edge!(g, 5, 2, Dict(:weight => 1, :density => 0.1))

    sys = system_from_graph(g; name=namespace=global_ns)
    prob = ODEProblem(sys, [], (0,2))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success 
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
end

@testset "Single Block" begin
    @named solo = JansenRit()
    g = MetaDiGraph()
    add_blox!(g, solo)
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

@testset "LIFExciBlox - LIFInhBlox network" begin
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
end

@testset "LIFExciCircuitBlox" begin
    @named n = LIFExciCircuitBlox(; N_neurons = 10, weight=1)

    sys_simpl = structural_simplify(n.system)
    prob = ODEProblem(sys_simpl, [], (0, 200.0))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success 
end

@testset "PoissonSpikeTrain - LIFExciBlox network" begin
    global_ns = :g # global namespace

    tspan = (0, 200) # ms
    spike_rate = 10* 1e-3 # spikes / ms

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns)
    @named n1 = LIFExciNeuron(; namespace = global_ns)

    neurons = [s, n1]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

@testset "PoissonSpikeTrain{<:Distribution} - LIFExciBlox network" begin
    global_ns = :g # global namespace

    tspan = (0, 200) # ms
    spike_rate = (distribution=Normal(3, 0.1), dt=10)

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns)
    @named n1 = LIFExciNeuron(; namespace = global_ns)

    neurons = [s, n1]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
end

@testset "PoissonSpikeTrain - LIFExciCircuitBlox" begin    
    global_ns = :g # global namespace

    tspan = (0, 1000) # ms
    spike_rate = 10* 1e-3 # spikes / ms

    @named s = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns)
    @named n = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = 10, weight=1)

    neurons = [s, n]

    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, 1, 2, Dict(:weight => 1))

    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success
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
    prob = SDEProblem(sys, [], (0.0, sim_dur))
    sol = solve(prob, RKMil(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success
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
    sol = solve(prob, RKMil(), saveat=1.0)
    @test sol.retcode == ReturnCode.Success
end

@testset "VdP" begin
    Random.seed!(1234)
    @testset "Non-noisy" begin
        @named vdp = VanDerPol()
        g = MetaDiGraph()
        add_blox!(g, vdp)
        @named sys = system_from_graph(g)
        prob = ODEProblem(sys, [0.0, 0.1], (0.0, 20.0), [])
        sol = solve(prob,Tsit5())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "Noisy" begin
        @named vdp = VanDerPol(include_noise=true)
        g = MetaDiGraph()
        add_blox!(g, vdp)
        @named sys = system_from_graph(g)
        prob = SDEProblem(sys, [0.0, 0.1], (0.0, 20.0), [])
        sol = solve(prob, RKMil())
        @test sol.retcode == ReturnCode.Success
    end
end

@testset "DBS circuit firing rates" begin
    @testset "Striatum_MSN_Adam" begin
        Random.seed!(1234)
        @named msn = Striatum_MSN_Adam(N_inhib = 10)
        sys = structural_simplify(get_system(msn))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(msn, ens_sol, threshold=-35, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 5.74, atol = 0.49)
    end
    @testset "Striatum_FSI_Adam" begin
        Random.seed!(1234)
        @named fsi = Striatum_FSI_Adam(N_inhib = 10)
        sys = structural_simplify(get_system(fsi))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(fsi, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 12.02, atol = 0.1)
    end
    @testset "GPe_Adam" begin
        Random.seed!(1234)
        @named gpe = GPe_Adam(N_inhib = 10, density = 0.5, weight = 0.5)
        sys = structural_simplify(get_system(gpe))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(gpe, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 32.46, atol = 0.18)
    end
    @testset "STN_Adam" begin
        Random.seed!(1234)
        @named stn = STN_Adam(N_exci = 10, density = 0.5, weight = 0.5)
        sys = structural_simplify(get_system(stn))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(stn, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 292.63, atol = 4.79)
    end

end

@testset "Single MetabolicHHNeuron" begin
    @named solo = MetabolicHHNeuron(neurontype=:excitatory, λ=1, τ=4, I_in=-4)
    g = MetaDiGraph()
    add_blox!(g, solo)
    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    @test sol.retcode == ReturnCode.Success

    # Extract voltage time-series
    V_ind = findfirst(x -> occursin("V(t)", string(x)), unknowns(sys))
    V_ts = sol[V_ind, :]
    
    # Check if time points where V(t) > 0 mV occur throughout the timeseries
    t_above = sol.t[V_ts .> 0]
    @test minimum(t_above) < 50 && maximum(t_above) > 180
    
    # Check if time points where V(t) < -50 mV occur throughout the timeseries
    t_below = sol.t[V_ts .< -50]
    @test minimum(t_below) < 50 && maximum(t_below) > 180

end
@testset "MetabolicHHNeuron Network" begin
    N_exc = 10;
    N_inh = 2;
    N = N_exc + N_inh;
    w = 1.;

    assembly = [];
    for i in 1:N_exc
        push!(assembly, MetabolicHHNeuron(name=Symbol("nrn$i"), λ=1, τ=4, I_in=-4,
            neurontype=:excitatory));
    end
    for i in 1+N_exc:N_exc+N_inh
        push!(assembly, MetabolicHHNeuron(name=Symbol("nrn$(i)"), λ=0.5, τ=8, I_in=-4,
            neurontype=:inhibitory));
    end

    adj_matrix = rand(N, N) .< 0.2;

    g = MetaDiGraph();
    add_blox!.(Ref(g), assembly);

    for i in 1:N
        for j in 1:N
            if adj_matrix[i, j]
                add_edge!(g, i, j, :weight, w);
            end
        end
    end

    @named sys = system_from_graph(g);

    prob = ODEProblem(sys, [], [0., 200.], []);
    sol = solve(prob, Tsit5());

    @test sol.retcode == ReturnCode.Success

    # Extract voltage time-series from an exc and an inh neuron
    V_inds = findall(x -> occursin("V(t)", string(x)), unknowns(sys))
    V_ts_exc = sol[V_inds[1], :]
    V_ts_inh = sol[V_inds[N_exc+1], :]

    # Check if time points where V(t) > 20 mV occur throughout the ts
    t_above = sol.t[V_ts_exc .> -20]
    @test_skip minimum(t_above) < 50 && maximum(t_above) > 150
    t_above = sol.t[V_ts_inh .> -20]
    @test_skip minimum(t_above) < 50 && maximum(t_above) > 150

    # Check if time points where V(t) < -40 mV occur throughout the ts
    t_below = sol.t[V_ts_exc .< -40]
    @test_skip minimum(t_below) < 50 && maximum(t_below) > 150
    t_below = sol.t[V_ts_inh .< -40]
    @test_skip minimum(t_below) < 50 && maximum(t_below) > 150

end
