using NeurobloxSynapticBlox, GraphDynamics, Test, Random, OrdinaryDiffEqVerner, OhMyThreads, Statistics
using DataFrames: DataFrames, DataFrame
using CSV: CSV
using Random
using Downloads

rrng() = Xoshiro(rand(Int))

function small_corticostriatal_learning_run(;time_block_dur = 90.0, ## ms (size of discrete time blocks)
                                            N_trials = 700, ## number of trials
                                            trial_dur = 1000, ## ms
                                            seed = nothing,
                                            graphdynamics=true,
                                            )
    if !isnothing(seed)
        Random.seed!(seed)
    end
    @time begin
        image_set = CSV.read(
            Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv"), DataFrame) ## reading data into DataFrame format
        model_name=:g
        ## define stimulus source blox
        ## t_stimulus: how long the stimulus is on (in msec)
        ## t_pause : how long th estimulus is off (in msec)
        @named stim = ImageStimulus(image_set; namespace=model_name, t_stimulus=trial_dur, t_pause=0); 

        ## cortical blox
        @named VAC = Cortical(; namespace=model_name, N_wta=4, N_exci=5,  density=0.05, weight=1, rng=rrng())
        @named AC = Cortical(; namespace=model_name, N_wta=2, N_exci=5, density=0.05, weight=1, rng=rrng()) 
        ## ascending system blox, modulating frequency set to 16 Hz
        @named ASC1 = NGNMM_theta(; namespace=model_name, Cₑ=2*26,Cᵢ=1*26, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑᵢ=0.6*26, kᵢₑ=0.6*26) 

        ## striatum blocks
        @named STR1 = Striatum(; namespace=model_name, N_inhib=5) 
        @named STR2 = Striatum(; namespace=model_name, N_inhib=5) 

        @named tan_pop1 = TAN(κ=10; namespace=model_name, rng=rrng()) 
        @named tan_pop2 = TAN(κ=10; namespace=model_name, rng=rrng()) 
	    
        @named AS = GreedyPolicy(; namespace=model_name, t_decision=2*time_block_dur) 
        @named SNcb = SNc(κ_DA=1; namespace=model_name) 

        # learning rules
        hebbian_mod = HebbianModulationPlasticity(K=0.06, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=trial_dur, t_post=trial_dur, t_mod=time_block_dur)
        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=7, t_pre=trial_dur, t_post=trial_dur)

        g = GraphSystem()
        add_connection!(g, stim => VAC, weight=14, rng=rrng())
        add_connection!(g, ASC1 => VAC, weight=44, rng=rrng())
        add_connection!(g, ASC1 => AC, weight=44, rng=rrng())
        add_connection!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort, rng=rrng())
        add_connection!(g, AC => STR1, weight = 0.075, density =  0.04, learning_rule =  hebbian_mod, rng=rrng())
        add_connection!(g, AC => STR2, weight =  0.075, density =  0.04, learning_rule =  hebbian_mod, rng=rrng())
        
        add_connection!(g, tan_pop1 => STR1, weight = 1, t_event = time_block_dur, rng=rrng())
        add_connection!(g, tan_pop2 => STR2, weight = 1, t_event = time_block_dur, rng=rrng())
        add_connection!(g, STR1 => tan_pop1, weight = 1, rng=rrng())
        add_connection!(g, STR2 => tan_pop1, weight = 1, rng=rrng())
        add_connection!(g, STR1 => tan_pop2, weight = 1, rng=rrng())
        add_connection!(g, STR2 => tan_pop2, weight = 1, rng=rrng())
        add_connection!(g, STR1 => STR2, weight = 1, t_event = 2*time_block_dur, rng=rrng())
        add_connection!(g, STR2 => STR1, weight = 1, t_event = 2*time_block_dur, rng=rrng())
        
        add_connection!(g, STR1 => SNcb, weight = 1, rng=rrng())
        add_connection!(g, STR2 => SNcb, weight = 1, rng=rrng())
        # action selection connections
        add_connection!(g, STR1 => AS);
        add_connection!(g, STR2 => AS);

        @named env = ClassificationEnvironment(stim, N_trials)
        @named agent = Agent(g; t_block = time_block_dur, graphdynamics);
        print("Construction:  "); 
    end
    trace = run_experiment!(agent, env; t_warmup=200.0, alg=Vern7(), monitor=ProgressMeterMonitor(N_trials), save_everystep=false)
end


function big_corticostriatal_learning_run(; t_block = 90, # ms (size of discrete time blocks)
                                          scheduler = SerialScheduler(),
                                          t_warmup=800.0,
                                          seed = nothing,
                                          graphdynamics=true,
                                          N_trials = 700, #number of trials
                                          )
    if !isnothing(seed)
        Random.seed!(seed)
    end
    @time begin
	    global_ns = :g 

	    fn = joinpath(@__DIR__, "../../examples/image_example.csv") #stimulus image file
        data = CSV.read(fn, DataFrame)

	    #define the circuit blox
        @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=600, t_pause=1000) 
	    
        @named LC = NGNMM_theta(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 
	    
        @named ITN = NGNMM_theta(;namespace=global_ns, Cₑ=2*36,Cᵢ=1*36, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/36, alpha_invₑᵢ=0.8/36, alpha_invᵢₑ=10.0/36, alpha_invᵢᵢ=0.8/36, kₑₑ=0.0*36, kₑᵢ=0.6*36, kᵢₑ=0.6*36, kᵢᵢ=0*36) 
	    
	    @named VC = Cortical(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0;namespace=global_ns, rng=rrng()) 

        @named PFC = Cortical(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0;namespace=global_ns, rng=rrng())
        
	    @named STR1 = Striatum(N_inhib=25;namespace=global_ns) 
	    @named STR2 = Striatum(N_inhib=25;namespace=global_ns) 
	    
	    @named tan_nrn = HHExci(;namespace=global_ns) 
	    
	    @named gpi1 = GPi(N_inhib=25;namespace=global_ns) 
	    @named gpi2 = GPi(N_inhib=25;namespace=global_ns) 
	    
	    @named gpe1 = GPe(N_inhib=15;namespace=global_ns) 
	    @named gpe2 = GPe(N_inhib=15;namespace=global_ns) 
	    
	    @named STN1 = STN(N_exci=15,I_bg=3*ones(25); namespace=global_ns) 
        @named STN2 = STN(N_exci=15,I_bg=3*ones(25); namespace=global_ns) 

        # @named Thal1 = Thalamus(N_exci=5;namespace=global_ns)
	    @named Thal1 = Thalamus(N_exci=25;namespace=global_ns) 
	    @named Thal2 = Thalamus(N_exci=25;namespace=global_ns) 
        
        @named tan_pop1 = TAN(;namespace=global_ns) 
        @named tan_pop2 = TAN(;namespace=global_ns) 
	    
	    @named AS = GreedyPolicy(namespace=global_ns, t_decision=180.5) 
        @named SNcb = SNc(namespace=global_ns)

	    #define learning rules
	    hebbian_mod = HebbianModulationPlasticity(K=0.04, decay=0.01, α=2.5, θₘ=1.0, modulator=SNcb, t_pre=1600-eps(), t_post=1600-eps(), t_mod=90.0)
	    
        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=5, t_pre=1600-eps(), t_post=1600-eps())
	    
	    hebbian_thal_cort = HebbianPlasticity(K=1.7e-5, W_lim=6, t_pre=1600-eps(), t_post=1600-eps())

	    g = GraphSystem()
        add_connection!(g, LC => VC; weight = 44, rng=rrng()) #LC->VC
	    add_connection!(g, LC => PFC; weight = 44, rng=rrng()) #LC->pfc
	    add_connection!(g, ITN => tan_nrn; weight = 100, rng=rrng()) #ITN->tan
        add_connection!(g, VC => PFC; weight = 0.1, density = 0.08, learning_rule = hebbian_cort, rng=rrng()) #VC->pfc
        add_connection!(g, PFC => STR1; weight = 0.03, density = 0.04, learning_rule = hebbian_mod, rng=rrng()) #pfc->str1
        add_connection!(g, PFC => STR2; weight = 0.03, density = 0.04, learning_rule = hebbian_mod, rng=rrng()) #pfc->str2
	    add_connection!(g, tan_nrn => STR1; weight = 0.17, rng=rrng()) #tan->str1
	    add_connection!(g, tan_nrn => STR2; weight = 0.17, rng=rrng()) #tan->str2
	    add_connection!(g, STR1 => gpi1, weight = 4, density = 0.04, rng=rrng()) #str1->gpi1
	    add_connection!(g, STR2 => gpi2; weight = 4, density = 0.04, rng=rrng()) #str2->gpi2
	    add_connection!(g, gpi1 => Thal1; weight = 0.16, density = 0.04, rng=rrng()) #gpi1->thal1
	    add_connection!(g, gpi2 => Thal2; weight = 0.16, density = 0.04, rng=rrng()) #gpi2->thal2
        add_connection!(g, Thal1 => PFC; weight = 0.2, density = 0.32, learning_rule = hebbian_thal_cort, sta=true, rng=rrng()) #thal1->pfc
	    add_connection!(g, Thal2 => PFC; weight = 0.2, density = 0.32, learning_rule = hebbian_thal_cort, sta=true, rng=rrng()) #thal2->pfc
	    add_connection!(g, STR1 => gpe1; weight = 4, density = 0.04, rng=rrng())   #str1->gpe1
	    add_connection!(g, STR2 => gpe2; weight = 4.0, density = 0.04, rng=rrng()) #str2->gpe2
	    add_connection!(g, gpe1 => gpi1; weight = 0.2, density = 0.04, rng=rrng()) #gpe1->gpi1
	    add_connection!(g, gpe2 => gpi2; weight = 0.2, density = 0.04, rng=rrng()) #gpe2->gpi2
	    add_connection!(g, gpe1 => STN1; weight = 3.5, density = 0.04, rng=rrng()) #gpe1->stn1
	    add_connection!(g, gpe2 => STN2; weight = 3.5, density = 0.04, rng=rrng()) #gpe2->stn2
	    add_connection!(g, STN1 => gpi1; weight = 0.1, density = 0.04, rng=rrng()) #stn1->gpi1
	    add_connection!(g, STN2 => gpi2; weight = 0.1, density = 0.04, rng=rrng()) #stn2->gpi2
	    add_connection!(g, stim => VC; weight = 14, rng=rrng()) #stim->VC
	    add_connection!(g, tan_pop1 => STR1; weight = 1, t_event = 90.0, rng=rrng()) #TAN pop1 -> str1
	    add_connection!(g, tan_pop2 => STR2; weight = 1, t_event = 90.0, rng=rrng()) #TAN pop2 -> str2
	    add_connection!(g, STR1 => tan_pop1; weight = 1, rng=rrng()) #str1 -> TAN pop1 
	    add_connection!(g, STR2 => tan_pop1; weight = 1, rng=rrng()) #str2 -> TAN pop1
	    add_connection!(g, STR1 => tan_pop2; weight = 1, rng=rrng()) #str1 -> TAN pop2 
	    add_connection!(g, STR2 => tan_pop2; weight = 1, rng=rrng()) #str2 -> TAN pop2
	    add_connection!(g, STR1 => STR2; weight = 1, t_event = 181.0, rng=rrng()) #str1 -> str2
	    add_connection!(g, STR2 => STR1; weight = 1, t_event = 181.0, rng=rrng()) #str2 -> str1
	    add_connection!(g, STR1 => AS, rng=rrng())# str1->AS
	    add_connection!(g, STR2 => AS, rng=rrng())# str2->AS
	    add_connection!(g, STR1 => SNcb; weight = 1.0, rng=rrng()) # str1->Snc
        add_connection!(g, STR2 => SNcb; weight = 1.0, rng=rrng())  # str2->Snc

        @named env = ClassificationEnvironment(stim, N_trials)
        @named agent = Agent(g; t_block, graphdynamics, scheduler);
        print("Construction:  ");
    end
    run_experiment!(agent, env;
                    alg=Vern7(),
                    t_warmup,
                    save_everystep=false,
                    monitor=ProgressMeterMonitor(N_trials),
                    modulator=SNcb)
end

function DA_tests(trace)
    # For the first 20 or so trials, DA is almost always 1.0 or 2.0 (incorrect or correct respectively)
    let N_under = 0
        for i ∈ 1:20
            if trace.correct[i]
                @test 1 < trace.DA[i] <= 2
                if trace.DA[i] < 2
                    N_under += 1
                end
            else
                @test 0 < trace.DA[i] <= 1
                if trace.DA[i] < 1
                    N_under += 1
                end
            end
        end
        @test N_under <= 5
    end
    # a bit after 20 trials, dopamine values start decreasing both for the correct and incorrect responses
    let N_under = 0
        for i ∈ 40:100
            if trace.correct[i]
                @test 1 < trace.DA[i] <= 2
                if trace.DA[i] < 2
                    N_under += 1
                end
            else
                @test 0 < trace.DA[i] <= 1
                if trace.DA[i] < 1
                    N_under += 1
                end
            end
        end
        @test N_under > 40
    end  
end
