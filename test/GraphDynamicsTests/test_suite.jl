using
    GraphDynamics,
    Test,
    OrdinaryDiffEq,
    Distributions,
    ModelingToolkit,
    Random,
    StochasticDiffEq,
    Neuroblox,
    Graphs,
    MetaGraphs,
    Random,
    LinearAlgebra

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using Neuroblox: NeuralMassBlox, AbstractNeuronBlox
using Base.Iterators: map as imap
using GraphDynamics.SymbolicIndexingInterface

function test_compare_du_and_sols(::Type{ODEProblem}, g, tspan;
                                  u0map=[], param_map=[],
                                  rtol,
                                  parallel=true, mtk=true, alg=nothing)
    if g isa Tuple
        (gl, gr) = g
    else
        gl = g
        gr = g
    end
    @named gsys = system_from_graph(gl; graphdynamics=true)
    state_names = variable_symbols(gsys)
    sol_grp, du_grp = let sys = gsys
        prob = ODEProblem(sys, u0map, tspan, param_map)
        (; f, u0, p) = prob
        du = similar(u0)
        f(du, u0, p, 1.0)

        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        sol_u_reordered = map(state_names) do name
            sol[name][end]
        end
        du_reordered = map(state_names) do name
            getu(sys, name)(du)
        end
        sol_u_reordered, du_reordered
    end
   
    if mtk
        sol_mtk, du_mtk = let @named sys = system_from_graph(gr)
            prob = ODEProblem(sys, u0map, tspan, param_map)
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)

            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            # For some reason getu is erroring here, this is some sort of MTK bug I think
            # du_reordered = map(state_names) do name 
            #     getu(sys, name)(du)
            # end
            du_reordered = du
            sol_u_reordered, du_reordered
        end
        @debug "" norm(sol_grp .- sol_mtk) / norm(sol_mtk)
        for i ∈ eachindex(state_names)
            if !isapprox(sol_grp[i], sol_mtk[i]; rtol=rtol)
                @debug  "" i state_names[i] sol_grp[i] sol_mtk[i]
            end
        end
        @test sort(du_grp) ≈ sort(du_mtk) # due to the MTK getu bug, we'll compare the sorted versions
        @test sol_grp ≈ sol_mtk rtol=rtol
    end
    if parallel
        sol_grp_p, du_grp_p = let sys = gsys
            prob = ODEProblem(sys, u0map, tspan, param_map, scheduler=StaticScheduler())
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)

            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            du_reordered = map(state_names) do name
                getu(sys, name)(du)
            end
            sol_u_reordered, du_reordered
        end
        @test du_grp ≈ du_grp_p
        @test sol_grp ≈ sol_grp_p rtol=rtol
    end
end

function basic_smoketest()
    Random.seed!(1234)
    @testset "Basic smoketest" begin
        #let
        # This is just some quick tests to hit some random mechanisms and make sure stuff at least runs before we move
        # on to tests that compare results from GraphDynamics against those from MTK.
        for (ProbType, alg, neurons) ∈ ((ODEProblem, Tsit5(), [IFNeuron(I_in=rand(), name=:lif1)
                                                               IFNeuron(I_in=rand(), name=:lif2)
                                                               QIFNeuron(I_in=rand(), name=:qif1)]),
                                        (SDEProblem, RKMil(), [HHNeuronInhib_GPe_Adam_Blox(name=:nrn1, I_bg=3, freq=4)
                                                               HHNeuronInhib_GPe_Adam_Blox(name=:nrn2, I_bg=2, freq=6)
                                                               HHNeuronExci_STN_Adam_Blox(name=:nrn3,  I_bg=2, freq=3)]))
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
                            if (neurons[i] isa NeuralMassBlox && neurons[j] isa AbstractNeuronBlox)
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

function basic_hh_network_tests()
    Random.seed!(1234)
    @testset "HH Neuron excitatory & inhibitory network" begin
        nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=3, freq=4)
        nn2 = HHNeuronExciBlox(name=Symbol("nrn2"), I_bg=2, freq=6)
        nn3 = HHNeuronInhibBlox(name=Symbol("nrn3"), I_bg=2, freq=3)
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

function vdp_test()
    @testset "VdP" begin
        Random.seed!(1234)
        @named vdp = VanDerPol()
        g = MetaDiGraph()
        add_blox!(g, vdp)
        test_compare_du_and_sols(ODEProblem, g, (0.0, 1.0); u0map=[vdp.x => 0.0, vdp.y=>0.1], rtol=1e-10, alg=Vern7())

        @named vdpn = VanDerPol(include_noise=true)
        @named vdpn2 = VanDerPol(include_noise=true)
        g = MetaDiGraph()
        add_blox!(g, vdpn)
        add_blox!(g, vdpn2)
        add_edge!(g, 1, 2, :weight, 1.0)
        
        prob = test_compare_du_and_sols(SDEProblem, g, (0.0, 1.0);
                                        u0map=[vdpn.x => 0.0, vdpn.y=>1.1], rtol=1e-10, alg=RKMil(), seed=123)
    end
end



function test_compare_du_and_sols_ensemble(::Type{SDEProblem}, graph, tspan; rtol, mtk=true, alg=nothing, trajectories=100_000)
    # Random.seed!(1234)
    if graph isa Tuple
        (graph_l, graph_r) = graph
    else
        graph_l = graph
        graph_r = graph
    end
    
    @named gsys = system_from_graph(graph_l; graphdynamics=true)
    state_names = variable_symbols(gsys)
    
    sol_grp_ens, du_grp, dnoise_grp = let sys = gsys
        prob = SDEProblem(sys, [], tspan, [])
        (; f, g, u0, p, noise_rate_prototype) = prob
        du = similar(u0)
        f(du, u0, p, 1.1)
        dnoise = zero(u0)
        g(dnoise, u0, p, 1.1)

        ens_prob = EnsembleProblem(prob)
        sols = solve(ens_prob, alg, EnsembleThreads(); trajectories)

        n_success = 0
        for sol ∈ sols
            n_success += sol.retcode == ReturnCode.Success
        end
        
        @test n_success > 0.95 * trajectories # allow up to 5% of the trajectories to fail
        sol_succ = [sol for sol in sols if sol.retcode == ReturnCode.Success]
        sols_u_reordered = map(sol_succ) do sol
            map(state_names) do name
                sol[name][end]
            end
        end
        sols_u_reordered, collect(du), collect(dnoise)
    end
    if mtk
        sol_mtk_ens, du_mtk, dnoise_mtk = let neuron_net = system_from_graph(graph_r; name=:neuron_net)
            prob = SDEProblem(neuron_net, [], tspan, [])
            (; f, g, u0, p, noise_rate_prototype) = prob
            du = similar(u0)
            f(du, u0, p, 1.1)

            dnoise = g(u0, p, 1.1)
            ens_prob = EnsembleProblem(prob)
            sols = solve(ens_prob, alg, EnsembleThreads(); trajectories)

            n_success = 0
            for sol ∈ sols
                n_success += sol.retcode == ReturnCode.Success
            end
            @test n_success > 0.95 * trajectories # allow up to 5% of the trajectories to fail
            sol_succ = [sol for sol in sols if sol.retcode == ReturnCode.Success]
            sols_u_reordered = map(sol_succ) do sol
                map(state_names) do name
                    sol[name][end]
                end
            end
            dnoise = sum(dnoise; dims=2)[:] # MTK doesn't understand that the noise is diagonal, so has a noise matrix instead
            sols_u_reordered, du, dnoise
        end
        @test sort(du_grp) ≈ sort(du_mtk)         #due to the MTK getu bug, we'll compare the sorted versions
        @test sort(dnoise_grp) ≈ sort(dnoise_mtk) #due to the MTK getu bug, we'll compare the sorted versions
        @debug "" norm(mean(sol_grp_ens) .- mean(sol_mtk_ens)) / norm(mean(sol_grp_ens))
        @test mean(sol_grp_ens) ≈ mean(sol_mtk_ens) rtol=rtol
        @test std(sol_grp_ens)  ≈ std(sol_mtk_ens)  rtol=rtol
    end
    nothing
end

function test_compare_du_and_sols(::Type{SDEProblem}, graph, tspan; rtol, mtk=true, alg=nothing, seed=1234,
                                  u0map=[], param_map=[],
                                  sol_comparison_broken=false, f_comparison_broken=false, g_comparison_broken=false)
    Random.seed!(seed)
    if graph isa Tuple
        (graph_l, graph_r) = graph
    else
        graph_l = graph
        graph_r = graph
    end
    @named gsys = system_from_graph(graph_l; graphdynamics=true)
    state_names = variable_symbols(gsys)
    sol_grp, du_grp, dnoise_grp = let sys = gsys
        prob = SDEProblem(sys, u0map, tspan, param_map, seed=seed)
        (; f, g, u0, p) = prob
        du = similar(u0)
        f(du, u0, p, 1.1)
        dnoise = zero(u0)
        g(dnoise, u0, p, 1.1)

        @test solve(prob, ImplicitEM(), saveat = 0.01,reltol=1e-4,abstol=1e-4).retcode == ReturnCode.Success
        
        sol = solve(prob, alg, saveat = 0.01)
        @test sol.retcode == ReturnCode.Success
        sol_reordered = map(state_names) do name
            sol[name][end]
        end
        sol_reordered, collect(du), collect(dnoise)
    end
    if mtk
        sol_mtk, du_mtk, dnoise_mtk = let neuron_net = system_from_graph(graph_r; name=:neuron_net)
            prob = SDEProblem(neuron_net, u0map, tspan, param_map, seed=seed)
            (; f, g, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.1)

            dnoise = g(u0, p, 1.1)
            dnoise = sum(dnoise; dims=2)[:] # MTK might not understand that the noise is diagonal, so it can give a diagonal matrix instead
            sol = solve(prob, alg, saveat = 0.01)
            @test sol.retcode == ReturnCode.Success
            sol_reordered = map(state_names) do name
                sol[name][end]
            end
            sol_reordered, collect(du), collect(dnoise)
        end
        @debug "" norm(sol_grp .- sol_mtk) / norm(sol_grp)
        #due to the MTK getu bug, we'll compare the sorted versions
        @test sort(du_grp) ≈ sort(du_mtk)         broken=f_comparison_broken   
        @test sort(dnoise_grp) ≈ sort(dnoise_mtk) broken=g_comparison_broken
        @test sol_grp ≈ sol_mtk rtol=rtol         broken=sol_comparison_broken
    end
    nothing
end

function stochastic_hh_network_tests()
    Random.seed!(1234)
    @testset "Adam_Brown_HH Neuron_network" begin
        nn1 = HHNeuronInhib_MSN_Adam_Blox(name=Symbol("nrn1"))
        nn2 = HHNeuronInhib_FSI_Adam_Blox(name=Symbol("nrn2"), σ=6)
        nn3 = HHNeuronInhib_FSI_Adam_Blox(name=Symbol("nrn3"), σ=6)
        nn4 = HHNeuronExci_STN_Adam_Blox(name=Symbol("nrn4"), σ=8)
        nn5 = HHNeuronInhib_GPe_Adam_Blox(name=Symbol("nrn5"),σ=8)
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
        @named n1 = HHNeuronInhib_FSI_Adam_Blox(σ=1)
        @named n2 = HHNeuronInhib_FSI_Adam_Blox(σ=2, freq=0.5)
        assembly = [n1, n2]

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, Dict(:weight=> 1.11, :gap => false, :gap_weight => 1.5))
        add_edge!(g, 2, 1, Dict(:weight=> 1.13, :gap => false, :gap_weight => 1.0))
        tspan = (0.0, 1.0)
        test_compare_du_and_sols(SDEProblem, g, tspan; rtol=1e-8, alg=RKMil())
    end
    @testset "FSI tests" begin
        @named n1 = HHNeuronInhib_FSI_Adam_Blox(σ=1)
        @named n2 = HHNeuronInhib_FSI_Adam_Blox(σ=2, freq=0.5)
        @named n3 = HHNeuronInhib_FSI_Adam_Blox(σ=3, freq=0.9)
        assembly = [n1, n2, n3]

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g, 1, 2, Dict(:weight=> 100.11, :gap => false, :gap_weight => 1.5))
        add_edge!(g, 1, 3, Dict(:weight=> 100.12, :gap => false))
        add_edge!(g, 2, 1, Dict(:weight=> 100.13, :gap => true, :gap_weight => 1.0))
        add_edge!(g, 2, 3, Dict(:weight=> 100.0,  :gap => true, :gap_weight => 1.0))
        # add_edge!(g, 3, 2, Dict(:weight=> 0.1,  :gap => true, :gap_weight=>1.0))
        # add_edge!(g, 2, 1, Dict(:weight=> 0.1, :gap => true, :gap_weight=>2.0))
        tspan = (0.0, 1.0)
        test_compare_du_and_sols(SDEProblem, g, tspan; rtol=1e-8, alg=RKMil())
    end
end


function ngei_test()
    @testset "NextGenerationEIBlox connected to neuron" begin
        global_ns = :g 
        @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
        @named nn = HHNeuronExciBlox(;namespace=global_ns)
        assembly = [LC, nn]
        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)
        add_edge!(g,1,2, :weight, 44)
        test_compare_du_and_sols(ODEProblem, g, (0.0, 2); rtol=1e-3)
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

function cortical_tests()
    Random.seed!(1234)
    @testset "Cortical blox" begin
        N_wta = 2
        N_exci = 5
        I_bg = [5 .* rand(N_exci) for _ ∈ 1:N_wta]
        weight = 1.0
        density = 0.5
        tspan = (0, 1.0)
        namespace = :g
        connection_matrices = map(Iterators.product(1:N_wta, 1:N_wta)) do _
            rand(Bernoulli(density), N_exci, N_exci)
        end
        g1 = let g = MetaDiGraph()
            @named cb = CorticalBlox(;N_wta, N_exci, I_bg_ar=I_bg, density, weight, namespace, connection_matrices)
            add_blox!(g, cb)
            g
        end
        g2 = let g = MetaDiGraph()
            @named cb = CorticalBlox(;N_wta, N_exci, I_bg_ar=I_bg, density, weight, namespace, connection_matrices)
            add_blox!(g, cb)
            g
        end
        test_compare_du_and_sols(ODEProblem, (g1, g2), tspan; rtol=1e-10, alg=Tsit5(), parallel=false)
    end
end


function striatum_tests(; sim_len=1.5)
    tspan = (0.0, sim_len)
    let g = MetaDiGraph()
        namespace = :g
        @named s1 = Striatum(;namespace)
        add_blox!(g, s1)
        test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-9, alg=Tsit5())
    end
    let g = MetaDiGraph()
        namespace = :g
        @named s1 = Striatum(;namespace, N_inhib=20, E_syn_inhib=-71.0, G_syn_inhib=1.3)
        @named s2 = Striatum(;namespace, N_inhib=30, E_syn_inhib=-69.0, G_syn_inhib=1.1)
        
        add_blox!(g, s1)
        add_blox!(g, s2)
        add_edge!(g, 1, 2, :t_event, 1.5)
        
        test_compare_du_and_sols(ODEProblem, g, tspan; rtol=1e-9, alg=Tsit5())
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
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            add_blox!(g, wta1)
            g
        end
        g2 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
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
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            @named wta3 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
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
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            @named wta3 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
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
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_edge!(g, 1, 2, Dict(:weight => weight, :connection_matrix => connection_matrix_1_2))
            g
        end
        g2 = let g = MetaDiGraph()
            @named wta1 = WinnerTakeAllBlox(;I_bg=I_bg_1, N_exci=N_exci_1, namespace)
            @named wta2 = WinnerTakeAllBlox(;I_bg=I_bg_2, N_exci=N_exci_2, namespace)
            add_blox!(g, wta1)
            add_blox!(g, wta2)
            add_edge!(g, 1, 2, Dict(:weight => weight, :connection_matrix => connection_matrix_1_2))
            g
        end
        test_compare_du_and_sols(ODEProblem, (g1, g2), tspan; rtol=1e-9, alg=Tsit5())
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
        make_conn = Neuroblox.indegree_constrained_connection_matrix
        
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

function discrete()
    @testset "Discrete blox" begin
        g = MetaDiGraph()
        @named n = HHNeuronExciBlox()
        @named m = Matrisome(t_event=8.0)
        @named t = TAN()
        add_blox!.((g,), (n, m, t))
        add_edge!(g, 1, 2, :weight, 1.0)
        add_edge!(g, 3, 2, Dict(:weight => 0.1, :t_event=>5.0))
        test_compare_du_and_sols(ODEProblem, g, (0.0, 20.0), rtol=1e-5, alg=Tsit5())
    end
end

function lif_exci_inh_tests(;tspan=(0.0, 20.0), rtol=1e-8)
    
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
    test_compare_du_and_sols(ODEProblem, g, tspan; rtol, alg=Tsit5())
end

function decision_making_test(;tspan=(0.0, 20.0), rtol=1e-5, N_E=24)
    
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

    @named n_A = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor);
    @named n_B = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) ;
    @named n_ns = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_nonselective, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
    @named n_inh = LIFInhCircuitBlox(; namespace = global_ns, N_neurons = N_I, weight = 1.0, exci_scaling_factor, inh_scaling_factor);

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

    test_compare_du_and_sols(ODEProblem, g, tspan; rtol, alg=Tsit5())
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
