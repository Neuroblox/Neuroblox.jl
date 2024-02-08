using Pkg
Pkg.activate(".")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("DifferentialEquations")


using Statistics
using CSV
using DataFrames
using Neuroblox
using Plots
using DifferentialEquations
using MetaGraphs

   
time_block_dur = 90 # ms (size of discrete time blocks)
N_trials = 700 #number of trials

	global_ns = :g 

	fn = joinpath(@__DIR__, "examples/image_example.csv") #stimulus image file
    data = CSV.read(fn, DataFrame)

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
	
	



#define the circuit as an Agent
  agent = Agent(g; name=:ag, t_block = 90);
  prob=agent.problem



	#extract membrane voltages of every neuron
    getsys=agent.odesystem;
	st=states(getsys)
	vlist=Int64[]
	for ii = 1:length(st)
		if contains(string(st[ii]), "V(t)")
			push!(vlist,ii)
		end
	end




#define environment : contains stimuli and feedback
	 env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
   





	t_warmup=800
    t_trial = env.t_trial
    tspan = (0, t_trial)
    stim_params_in = Neuroblox.get_trial_stimulus(env)
    sys = Neuroblox.get_sys(agent)
    prob2 = remake(prob; p = merge(stim_params_in),tspan=(0,1600))

	if t_warmup > 0
        prob2 = remake(prob2; tspan=(0,t_warmup))
        sol = solve(prob2, Vern7())
		u0 = sol[1:end,end] # last value of state vector
        prob2 = remake(prob2; p = merge(stim_params_in),tspan=tspan, u0=u0)
    else
        prob2 = remake(prob3; p = merge(stim_params_in),tspan=tspan)
        u0 = []
    end

    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    
    defs = ModelingToolkit.get_defaults(sys)
    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

	perf=zeros(N_trials)	
	act=zeros(N_trials)





		
    for ii = 1:N_trials
        prob3 = agent.problem
        stim_params = Neuroblox.get_trial_stimulus(env)
        prob3 = remake(prob3; p = merge(weights, stim_params), u0 = u0,tspan=(0,1600))
        @info env.current_trial
        sol2 = solve(prob3, Vern7())
        agent.problem = prob3
      
        if isnothing(action_selection)
            feedback = 1
        else
            action = action_selection(sol2)
			feedback = env(action)
			@info action
			@info env.category[env.current_trial]
	        @info feedback
			perf[ii]=feedback
			act[ii] = action
        end
     
        for (w, rule) in learning_rules
            w_val = weights[w]
            Δw = Neuroblox.weight_gradient(rule, sol2, w_val, feedback)
            weights[w] += Δw
        end
        Neuroblox.increment_trial!(env)

		
    end
	



Gray.(act/2)


Gray.(perf)


env.current_trial



	perf_smth=zeros(length(perf)-19)
	for ii=20:length(perf)
		perf_smth[ii-19] = mean(perf[ii-19:ii])
		
	end
	plot(collect(20:500),perf_smth[1:481],ylims=(0,1),xlims=(0,500))


