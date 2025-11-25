using NeurobloxPharma
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using ReferenceTests, CairoMakie
using StableRNGs

rng = StableRNG(2026)

@testset "HH Neuron excitatory & inhibitory network" begin
    nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=3)
    nn2 = HHNeuronExci(name=Symbol("nrn2"), I_bg=2)
    nn3 = HHNeuronInhib(name=Symbol("nrn3"), I_bg=2)
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

    plt = stackplot(assembly, sol)
    @test_reference "plots/hh_network.png" plt by=psnr_equality(40)
end

@testset "NGNMM_theta connected to neuron" begin
    global_ns = :g 
    @named LC = NextGenerationEI(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
    @named nn = HHNeuronExci(;namespace=global_ns)
    assembly = [LC, nn]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(neuron_net, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test neuron_net isa ODESystem
    @test sol.retcode == ReturnCode.Success

    plt = stackplot([nn], sol)
    @test_reference "plots/ngnmm_neuron.png" plt by=psnr_equality(40)
end

@testset "NGNMM_theta connected to Cortical" begin
    global_ns = :g 
    @named LC = NGNMM_theta(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
    @named cb = Cortical(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    assembly = [LC, cb]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(neuron_net, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success

    plt = meanfield(cb, sol)
    @test_reference "plots/ngnmm_cortical.png" plt by=psnr_equality(40)
end

@testset "WinnerTakeAll" begin
    N_exci = 5
    @named wta= WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci)
    sys = wta.system
    wta_simp=structural_simplify(sys)
    prob = ODEProblem(wta_simp,[],(0,10))
    sol = solve(prob, Vern7(), saveat=0.1)

    @test wta_simp isa ODESystem
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(wta, sol)
    @test_reference "plots/wta.png" plt by=psnr_equality(40)
end

@testset "WinnerTakeAll network" begin
    global_ns = :g # global namespace
    N_exci = 5
    @named wta1 = WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci, namespace=global_ns)
    @named wta2 = WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci, namespace=global_ns)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [wta1, wta2])
    add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.5, :rng => rng))
    sys = system_from_graph(g; name=global_ns)

    prob = ODEProblem(sys, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot([wta1, wta2], sol)
    @test_reference "plots/wta_network.png" plt by=psnr_equality(40)
end

@testset "Cortical" begin
    @named cb = Cortical(N_wta=6, N_exci=5, density=0.1, weight=1)
    cb_simpl = structural_simplify(cb.system)
    prob = ODEProblem(cb_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = meanfield(cb, sol)
    @test_reference "plots/cortical.png" plt by=psnr_equality(40)
end

@testset "Striatum" begin
    @named str_scb = Striatum(N_inhib=2)
    str_simpl = structural_simplify(str_scb.system)
    prob = ODEProblem(str_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = meanfield(str_scb, sol)
    @test_reference "plots/striatum.png" plt by=psnr_equality(40)
end

@testset "GPi" begin
    @named gpi_scb = GPi(N_inhib=2)
    gpi_simpl = structural_simplify(gpi_scb.system)
    prob = ODEProblem(gpi_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(gpi_scb, sol)
    @test_reference "plots/gpi.png" plt by=psnr_equality(40)
end

@testset "GPe" begin
    @named gpe_scb = GPe(N_inhib=2)
    gpe_simpl = structural_simplify(gpe_scb.system)
    prob = ODEProblem(gpe_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(gpe_scb, sol)
    @test_reference "plots/gpe.png" plt by=psnr_equality(40)
end

@testset "STN" begin
    @named stn_scb = STN(N_exci=2)
    stn_simpl = structural_simplify(stn_scb.system)
    prob = ODEProblem(stn_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(stn_scb, sol)
    @test_reference "plots/stn.png" plt by=psnr_equality(40)
end

@testset "Thalamus" begin
    @named thal_scb = Thalamus(N_exci=2)
    thal_simpl = structural_simplify(thal_scb.system)
    prob = ODEProblem(thal_simpl, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(thal_scb, sol)
    @test_reference "plots/thalamus.png" plt by=psnr_equality(40)
end

@testset "Cortical-ImageStimulus network" begin
    global_ns = :g # global namespace
    @named cb = Cortical(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    fn = joinpath(@__DIR__, "../../examples/image_example.csv")
    @named stim = ImageStimulus(fn; namespace=global_ns, t_stimulus=1, t_pause=0.5)
    g = MetaDiGraph()
    add_blox!(g, stim)
    add_blox!(g, cb)
    add_edge!(g, 1, 2, :weight, 1)
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], (0, 2))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(cb, sol)
    @test_reference "plots/cortical_image.png" plt by=psnr_equality(40)
end

@testset "Cortical-Cortical network" begin
    global_ns = :g # global namespace
    @named cb1 = Cortical(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    @named cb2 = Cortical(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    g = MetaDiGraph()
    add_blox!.(Ref(g), [cb1, cb2])
    add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.1))
    sys = system_from_graph(g; name=global_ns, t_block=90.0)
    prob = ODEProblem(sys, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot([cb1, cb2], sol)
    @test_reference "plots/cort_cort.png" plt by=psnr_equality(40)
end

@testset "Cortical & subcortical components network" begin
    global_ns = :g # global namespace
    @named cb1 = Cortical(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named cb2 = Cortical(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
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

    fig, ax, plt = meanfield(cb1, sol)
    meanfield!(ax, cb2, sol)
    meanfield!(ax, str1, sol)
    meanfield!(ax, gpi1, sol)
    meanfield!(ax, thal1, sol)
    @test_reference "plots/cort_subcort.png" fig by=psnr_equality(40)
end
