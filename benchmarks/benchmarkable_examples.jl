#------------------------------------------------------------------------------
# Loading Neuroblox
#------------------------------------------------------------------------------

load_neuroblox = BenchSetup(
    name = "Load NeuroBlox",
    bench = :(@eval using Neuroblox)
)

#------------------------------------------------------------------------------
# Simple neuron creation
#------------------------------------------------------------------------------

linear_neural_mass_creation = BenchSetup(
    name = "Create LinearNeuralMass",
    setup = :(using Neuroblox),
    bench = :(@named lm1 = LinearNeuralMass())
)

#------------------------------------------------------------------------------
# Harmonic Oscillator
#------------------------------------------------------------------------------

harmonic_oscillator_system_creation = BenchSetup(
    name = "Harmonic Oscillator system creation",
    setup = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
    bench = quote
        @named osc1 = HarmonicOscillator()
        @named osc2 = HarmonicOscillator()

        params = @parameters k=1.0
        adj = [0 k; k 0]
        
        g = MetaDiGraph()
        add_blox!.(Ref(g), [osc1, osc2])
        create_adjacency_edges!(g, adj)
        @named sys = system_from_graph(g, Num[])
    end
)

harmonic_oscillator_structural_simplify = BenchSetup(
    name = "Harmonic Oscillator structural simplify",
    setup = quote
        $(harmonic_oscillator_system_creation.setup) # re-use the previous setup
        $(harmonic_oscillator_system_creation.bench)   # re-use the previous bench as a setup
    end,
    args = [:sys],
    bench = :(final_sys = structural_simplify(sys))
)

harmonic_oscillator_solve = BenchSetup(
    name="Harmonic Oscillator solve",
    setup=quote
        $(harmonic_oscillator_structural_simplify.setup) # re-use the previous setup
        $(harmonic_oscillator_structural_simplify.bench) # re-use the previous benchmark as a setup
        sim_dur = 20_000.0 # 20 seconds
        prob = ODEProblem(final_sys, [], (0.0, sim_dur),[])
    end,
    args = [:prob],
    bench = :(sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)),
    seconds = 50
)

#------------------------------------------------------------------------------
# Jansen Ritt
#------------------------------------------------------------------------------

jansen_ritt_system_creation = BenchSetup(
    name="Jansen-Ritt system creation",
    setup = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
    bench = quote
        @named GPe       = JansenRit(τ=40.0, H=20, λ=400, r=0.1)
        @named STN       = JansenRit(τ=10.0, H=20, λ=500, r=0.1)
        @named GPi       = JansenRit(τ=14.0, H=20, λ=400, r=0.1)
        @named Thalamus  = JansenRit(τ=2.0, H=10, λ=20, r=5)
        @named PFC       = JansenRit(τ=1.0, H=20, λ=5, r=0.15)
        blox = [GPe, STN, GPi, Thalamus, PFC]
        
        # Store parameters to be passed later on
        params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5
        
        g = MetaDiGraph()
        add_blox!.(Ref(g), blox)
        
        add_edge!(g,1,1, Dict(:weight => -0.5*C_BG_Th, :delay => 0.1))
        add_edge!(g,1,2, Dict(:weight => C_BG_Th,      :delay => 0.2))
        add_edge!(g,2,1, Dict(:weight => -0.5*C_BG_Th, :delay => 0.3))
        add_edge!(g,2,5, Dict(:weight => C_Cor_BG_Th,  :delay => 0.4))
        add_edge!(g,3,1, Dict(:weight => -0.5*C_BG_Th, :delay => 0.5))
        add_edge!(g,3,2, Dict(:weight => C_BG_Th,      :delay => 0.6))
        add_edge!(g,4,3, Dict(:weight => -0.5*C_BG_Th, :delay => 0.7))
        add_edge!(g,4,4, Dict(:weight => C_BG_Th_Cor,  :delay => 0.8))

        sys_delays = graph_delays(g)
        @named sys = system_from_graph(g, params)
    end
)

jansen_ritt_structural_simplify = BenchSetup(
    name="Jansen-Ritt structural simplify",
    setup = quote
        $(jansen_ritt_system_creation.setup)
        $(jansen_ritt_system_creation.bench)
    end,
    args = [:sys],
    bench = :(final_sys = structural_simplify(sys))
)

jansen_ritt_solve = BenchSetup(
    name="Jansen-Ritt solve",
    setup = quote
        $(jansen_ritt_structural_simplify.setup) # re-use the previous setup
         $(jansen_ritt_structural_simplify.bench) # re-use the previous benchmark as a setup

        sim_dur = 20_000.0 # Simulate for 20 Seconds

        # Jansen-Rit allows delays and therefore we create a delayed
        # differential equation problem
        prob = DDEProblem(final_sys,
                          [],
                          (0.0, sim_dur),
                          constant_lags = sys_delays)
        alg = MethodOfSteps(Vern7())
        
    end,
    args = [:prob],
    bench= :(solve(prob, MethodOfSteps(Vern7()), saveat=1))
)

#------------------------------------------------------------------------------
# 40 Random Neurons
#------------------------------------------------------------------------------

forty_neurons_system_creation = BenchSetup(
    name="40 randomly connected neurons system creation",
    setup = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
    bench = let N = 40, rng = Xoshiro(seed)
        neuron_names = [Symbol(:neuron,i) for i ∈ 1:N]
        neurons = map(1:N) do i
            type = rand(rng, (:HarmonicOscillator, :LinearNeuralMass, :WilsonCowan))
            :(@named $(neuron_names[i]) = $type())
        end
        adj = randn(rng, N, N)
        for i ∈ 1:N
            for j ∈ 1:N
                if rand(rng) < 0.20
                    adj[i, j] = 0
                end
            end
        end
        quote
            $(Expr(:block, neurons...))
            adj = $adj
            g = MetaDiGraph()
            add_blox!.(Ref(g), [$(neuron_names...)])
            create_adjacency_edges!(g, adj)
            @named sys = system_from_graph(g, Num[])
        end
    end
)

forty_neurons_structural_simplify = BenchSetup(
    name="40 randomly connected neurons structural simplify",
    setup = quote
        using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
        $(forty_neurons_system_creation.setup)
        $(forty_neurons_system_creation.bench)
    end,
    args = [:sys,],
    bench = :(final_sys = structural_simplify(sys))
)

forty_neurons_solve = BenchSetup(
    name="40 randomly connected neurons solve",
    setup = quote
        using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
        $(forty_neurons_structural_simplify.setup)
        $(forty_neurons_structural_simplify.bench)
        sim_dur = 50.0
        prob = ODEProblem(final_sys, [], (0.0, sim_dur),[])
    end,
    args = [:prob,],
    bench = :(solve(prob, Vern7()# , saveat=0.1
                    ))
)

#------------------------------------------------------------------------------
# RF Learning
#------------------------------------------------------------------------------

rf_learning_setup = BenchSetup(
    name="RF_learning_simple agent and envionment creation",
    setup=quote
        using CSV,DataFrames, Neuroblox, DifferentialEquations, MetaGraphs
        
        time_block_dur = 90 # ms (size of discrete time blocks)
        N_trials = 3 #number of trials
        
        global_ns = :g
        
        fn = joinpath(@__DIR__(), "..", "examples", "image_example.csv") #stimulus image file
        data = CSV.read(fn, DataFrame)
    end,
    args = [:time_block_dur, :N_trials, :global_ns, :data],
    bench=quote
        #define the circuit blox
        @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=600, t_pause=1000) 

        @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 
        
        @named VC = CorticalBlox(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) 

        @named PFC = CorticalBlox(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) 

        @named STR1 = Striatum(N_inhib=1;namespace=global_ns) 
        @named STR2 = Striatum(N_inhib=1;namespace=global_ns) 

        @named tan_pop1 = TAN(;namespace=global_ns) 
        @named tan_pop2 = TAN(;namespace=global_ns) 

        @named AS = GreedyPolicy(namespace=global_ns, t_decision=180.5)
        @named SNcb = SNc(namespace=global_ns)

        assembly = [LC, VC, PFC, STR1, STR2, stim, tan_pop1, tan_pop2, AS, SNcb]
        d = Dict(b => i for (i,b) in enumerate(assembly))

        #define learning rules
        hebbian_mod = HebbianModulationPlasticity(K=0.04, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=1600-eps(), t_post=1600-eps(), t_mod=90)

        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=5, t_pre=1600-eps(), t_post=1600-eps())

        hebbian_thal_cort = HebbianPlasticity(K=1.7e-5, W_lim=6, t_pre=1600-eps(), t_post=1600-eps())

        g = MetaDiGraph()
        add_blox!.(Ref(g), assembly)

        #connect the bloxs

        add_edge!(g, d[LC], d[VC], Dict(:weight => 44)) #LC->VC
        add_edge!(g, d[LC], d[PFC], Dict(:weight => 44)) #LC->pfc

        add_edge!(g, d[VC], d[PFC], Dict(:weight => 1, :density => 0.08, :learning_rule => hebbian_cort)) #VC->pfc
        add_edge!(g, d[PFC], d[STR1], Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str1
        add_edge!(g, d[PFC], d[STR2], Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str2
        
        add_edge!(g, d[stim], d[VC], :weight, 14) #stim->VC
        add_edge!(g, d[tan_pop1], d[STR1], Dict(:weight => 1, :t_event => 90.0)) #TAN pop1 -> str1
        add_edge!(g, d[tan_pop2], d[STR2], Dict(:weight => 1, :t_event => 90.0)) #TAN pop2 -> str2
        add_edge!(g, d[STR1], d[tan_pop1], Dict(:weight => 1)) #str1 -> TAN pop1 
        add_edge!(g, d[STR2], d[tan_pop1], Dict(:weight => 1)) #str2 -> TAN pop1
        add_edge!(g, d[STR1], d[tan_pop2], Dict(:weight => 1)) #str1 -> TAN pop2 
        add_edge!(g, d[STR2], d[tan_pop2], Dict(:weight => 1)) #str2 -> TAN pop2
        add_edge!(g, d[STR1], d[STR2], Dict(:weight => 1, :t_event => 181.0)) #str1 -> str2
        add_edge!(g, d[STR2], d[STR1], Dict(:weight => 1, :t_event => 181.0)) #str2 -> str1
        add_edge!(g, d[STR1], d[AS])# str1->AS
        add_edge!(g, d[STR2], d[AS])# str2->AS
        add_edge!(g, d[STR1], d[SNcb], Dict(:weight => 1)) # str1->Snc
        add_edge!(g, d[STR2], d[SNcb], Dict(:weight => 1))  # str2->Snc

        agent = Agent(g; name=:ag, t_block = 90)
        env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    end
)

rf_learning_three_trials = BenchSetup(
    name="RF_learning_simple run_experiment (three trials)",
    setup=quote
        $(rf_learning_setup.setup) # re-use the setup
        $(rf_learning_setup.bench) # re-use the benchmark for this setup
    end,
    # Setup code is run before each benchmark sample
    sample_setup = quote
        reset!(agent)
        reset!(env)
    end,
    args = [:agent, :env],
    bench = :(run_experiment!(agent, env; alg=Vern7(), reltol=1e-9, abstol=1e-9)),
    # This has to be 1 otherwise we'd get errors since the `setup` code only runs once
    # per benchmark-sample, not once per eval in the sample:
    evals = 1,
    seconds=5_000
)
