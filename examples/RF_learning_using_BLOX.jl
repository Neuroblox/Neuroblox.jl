### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 7d500a22-c5ed-48d1-b009-4d4a0f03fc2a
using Pkg

# ╔═╡ 98f74502-74e6-11ee-34ff-7b16a35b2860
Pkg.activate(".")

# ╔═╡ 0e839efb-a15b-410b-93e7-a8668f515423
Pkg.add("CSV")

# ╔═╡ 7067bfc7-0e4f-4474-b79e-403e67bfeed8
Pkg.add("DataFrames")

# ╔═╡ 038809e1-7512-4f5d-8fbf-feef2cdce117
Pkg.add("Plots")

# ╔═╡ b4e8ba6e-e91a-474c-9314-8dd75fa2e82e
Pkg.add("DifferentialEquations")

# ╔═╡ b6a6209f-97f3-4844-928a-948f3d53b4d7
using Statistics

# ╔═╡ 1f184999-9dd8-4958-8158-157b683b6f5c
using CSV

# ╔═╡ 536b8983-8548-4144-be00-4d516fe7b75f
using DataFrames

# ╔═╡ e6a453f3-e2a8-44ec-9be2-1f6d9293c911
using Neuroblox

# ╔═╡ e8deef43-7ee9-484f-a8a8-2df6e7006e22
using Plots

# ╔═╡ ee0c04b9-57c9-44bc-b092-218019a545ec
using DifferentialEquations

# ╔═╡ d37ba3dd-78df-4e18-ac76-3864a2e9216e
using MetaGraphs

# ╔═╡ 8abc4a4b-4acf-4c25-a10c-33278329c9f4
begin

   
   time_block_dur = 90 # ms (size of discrete time blocks)
   N_trials = 700 #number of trials
	
	global_ns = :g 

	fn = joinpath(@__DIR__, "image_example.csv") #stimulus image file
    data = CSV.read(fn, DataFrame)

	#define the circuit blox
    @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=600, t_pause=1000) 
	
    @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 
	
   @named ITN = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*36,Cᵢ=1*36, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/36, alpha_invₑᵢ=0.8/36, alpha_invᵢₑ=10.0/36, alpha_invᵢᵢ=0.8/36, kₑₑ=0.0*36, kₑᵢ=0.6*36, kᵢₑ=0.6*36, kᵢᵢ=0*36) 
	
	@named VC = CorticalBlox(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) 

    @named PFC = CorticalBlox(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) 
 
	@named STR1 = Striatum(N_inhib=25;namespace=global_ns) 
	@named STR2 = Striatum(N_inhib=25;namespace=global_ns) 
	
	@named tan_nrn = HHNeuronExciBlox(;namespace=global_ns) 
	
	@named gpi1 = GPi(N_inhib=25;namespace=global_ns) 
	@named gpi2 = GPi(N_inhib=25;namespace=global_ns) 
	
	@named gpe1 = GPe(N_inhib=15;namespace=global_ns) 
	@named gpe2 = GPe(N_inhib=15;namespace=global_ns) 
	
	@named STN1 = STN(N_exci=15,I_bg=3*ones(25);namespace=global_ns) 
    @named STN2 = STN(N_exci=15,I_bg=3*ones(25);namespace=global_ns) 
	
	@named Thal1 = Thalamus(N_exci=25;namespace=global_ns) 
	@named Thal2 = Thalamus(N_exci=25;namespace=global_ns) 

    
    @named tan_pop1 = TAN(;namespace=global_ns) 
    @named tan_pop2 = TAN(;namespace=global_ns) 
	
	@named AS = GreedyPolicy(namespace=global_ns, t_decision=180.5) 
    @named SNcb = SNc(namespace=global_ns) 
	
	
    assembly = [LC, ITN, VC, PFC, STR1, STR2, tan_nrn, gpi1, gpi2, gpe1, gpe2, STN1, STN2, Thal1, Thal2, stim, tan_pop1, tan_pop2, AS, SNcb]
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
	add_edge!(g, d[ITN], d[tan_nrn], Dict(:weight => 100)) #ITN->tan
	add_edge!(g, d[VC], d[PFC], Dict(:weight => 1, :density => 0.08, :learning_rule => hebbian_cort)) #VC->pfc
	add_edge!(g, d[PFC], d[STR1], Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str1
	add_edge!(g, d[PFC], d[STR2], Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str2
	add_edge!(g, d[tan_nrn], d[STR1], Dict(:weight => 0.17)) #tan->str1
	add_edge!(g, d[tan_nrn], d[STR2], Dict(:weight => 0.17)) #tan->str2
	add_edge!(g, d[STR1], d[gpi1], Dict(:weight => 4, :density => 0.04)) #str1->gpi1
	add_edge!(g, d[STR2], d[gpi2], Dict(:weight => 4, :density => 0.04)) #str2->gpi2
	add_edge!(g, d[gpi1], d[Thal1], Dict(:weight => 0.16, :density => 0.04)) #gpi1->thal1
	add_edge!(g, d[gpi2], d[Thal2], Dict(:weight => 0.16, :density => 0.04)) #gpi2->thal2
    add_edge!(g, d[Thal1], d[PFC], Dict(:weight => 0.2, :density => 0.32, :learning_rule => hebbian_thal_cort, :sta => true)) #thal1->pfc
	add_edge!(g, d[Thal2], d[PFC], Dict(:weight => 0.2, :density => 0.32, :learning_rule => hebbian_thal_cort, :sta => true)) #thal2->pfc
	add_edge!(g, d[STR1], d[gpe1], Dict(:weight => 4, :density => 0.04)) #str1->gpe1
	add_edge!(g, d[STR2], d[gpe2], Dict(:weight => 4, :density => 0.04)) #str2->gpe2
	add_edge!(g, d[gpe1], d[gpi1], Dict(:weight => 0.2, :density => 0.04)) #gpe1->gpi1
	add_edge!(g, d[gpe2], d[gpi2], Dict(:weight => 0.2, :density => 0.04)) #gpe2->gpi2
	add_edge!(g, d[gpe1], d[STN1], Dict(:weight => 3.5, :density => 0.04)) #gpe1->stn1
	add_edge!(g, d[gpe2], d[STN2], Dict(:weight => 3.5, :density => 0.04)) #gpe2->stn2
	add_edge!(g, d[STN1], d[gpi1], Dict(:weight => 0.1, :density => 0.04)) #stn1->gpi1
	add_edge!(g, d[STN2], d[gpi2], Dict(:weight => 0.1, :density => 0.04)) #stn2->gpi2
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
	
	
end

# ╔═╡ 5ddd418f-23d2-4151-82c0-e8c9fae7e4f2
#define the circuit as an Agent
  agent = Agent(g; name=:ag, t_block = 90);

# ╔═╡ 7576c26d-8617-4c66-ab5e-db0bd8bb9e17
    prob=agent.problem

# ╔═╡ d80b3bf7-24d8-4395-8903-de974e6445f9
begin
	#extract membrane voltages of every neuron
    getsys=agent.system;
	st=unknowns(getsys)
	vlist=Int64[]
	for ii = 1:length(st)
		if contains(string(st[ii]), "V(t)")
			push!(vlist,ii)
		end
	end
end

# ╔═╡ bf1f09d9-7f9e-48f7-b3de-9fecae82a5a0
begin
#define environment : contains stimuli and feedback
	 env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
   
end

# ╔═╡ 14e2806e-5cdd-4b0b-b8c7-8c6d2c7e349c
begin 

	t_warmup=800
    t_trial = env.t_trial
    tspan = (0, t_trial)
    stim_params_in = Neuroblox.get_trial_stimulus(env)
    sys = Neuroblox.get_system(agent)
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
end


# ╔═╡ b7e84b20-0b80-478c-bc88-2883f80bcbb4
begin
	# sys = Neuroblox.get_system(agent)
	# learning_rules = agent.learning_rules
	# weights = Dict{Num, Float64}()
	# for w in keys(learning_rules)
	# 	weights[w] = defs[w]
	# end

    for ii = 1:N_trials
        prob3 = agent.problem
        stim_params = Neuroblox.get_trial_stimulus(env)
		new_params = ModelingToolkit.MTKParameters(sys, merge(defs, weights, stim_params))
        prob3 = remake(prob3; p = new_params, u0=u0, tspan=(0,env.t_trial))
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
	
end

# ╔═╡ 9703f279-a311-4aa2-85e4-75d8898b21b6
Gray.(act/2)

# ╔═╡ a973a05c-e686-48c5-8242-688e6475624b
Gray.(perf)

# ╔═╡ 35891714-fafc-4465-a2e5-bcd7aaf1a59c
begin
	perf_smth=zeros(length(perf)-19)
	for ii=20:length(perf)
		perf_smth[ii-19] = mean(perf[ii-19:ii])
		
	end
	plot(collect(20:500),perf_smth[1:481],ylims=(0,1),xlims=(0,500))
end

# ╔═╡ Cell order:
# ╠═7d500a22-c5ed-48d1-b009-4d4a0f03fc2a
# ╠═98f74502-74e6-11ee-34ff-7b16a35b2860
# ╠═b6a6209f-97f3-4844-928a-948f3d53b4d7
# ╠═0e839efb-a15b-410b-93e7-a8668f515423
# ╠═1f184999-9dd8-4958-8158-157b683b6f5c
# ╠═7067bfc7-0e4f-4474-b79e-403e67bfeed8
# ╠═536b8983-8548-4144-be00-4d516fe7b75f
# ╠═e6a453f3-e2a8-44ec-9be2-1f6d9293c911
# ╠═038809e1-7512-4f5d-8fbf-feef2cdce117
# ╠═e8deef43-7ee9-484f-a8a8-2df6e7006e22
# ╠═b4e8ba6e-e91a-474c-9314-8dd75fa2e82e
# ╠═ee0c04b9-57c9-44bc-b092-218019a545ec
# ╠═d37ba3dd-78df-4e18-ac76-3864a2e9216e
# ╠═8abc4a4b-4acf-4c25-a10c-33278329c9f4
# ╠═5ddd418f-23d2-4151-82c0-e8c9fae7e4f2
# ╠═7576c26d-8617-4c66-ab5e-db0bd8bb9e17
# ╠═d80b3bf7-24d8-4395-8903-de974e6445f9
# ╠═bf1f09d9-7f9e-48f7-b3de-9fecae82a5a0
# ╠═14e2806e-5cdd-4b0b-b8c7-8c6d2c7e349c
# ╠═b7e84b20-0b80-478c-bc88-2883f80bcbb4
# ╠═9703f279-a311-4aa2-85e4-75d8898b21b6
# ╠═a973a05c-e686-48c5-8242-688e6475624b
# ╠═35891714-fafc-4465-a2e5-bcd7aaf1a59c
