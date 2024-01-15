### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 7d500a22-c5ed-48d1-b009-4d4a0f03fc2a
using Pkg

# ╔═╡ 98f74502-74e6-11ee-34ff-7b16a35b2860
Pkg.activate(".")

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

   #t_trial = 1600 # ms
   time_block_dur = 90 # ms
    N_trials = 500
	
	global_ns = :g 

		fn = "examples/image_example.csv"
	
    @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) #1

	
	@named ITN = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*36,Cᵢ=1*36, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/36, alpha_invₑᵢ=0.8/36, alpha_invᵢₑ=10.0/36, alpha_invᵢᵢ=0.8/36, kₑₑ=0.0*36, kₑᵢ=0.6*36, kᵢₑ=0.6*36, kᵢᵢ=0*36) #2

	@named VC = CorticalBlox(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) #3
	
    @named PFC = CorticalBlox(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0;namespace=global_ns) #4

	

	@named STR1 = Striatum(N_inhib=25;namespace=global_ns) #5
	@named STR2 = Striatum(N_inhib=25;namespace=global_ns) #6
	@named tan_nrn = HHNeuronExciBlox(;namespace=global_ns) #7
	@named gpi1 = GPi(N_inhib=25;namespace=global_ns) #8
	@named gpi2 = GPi(N_inhib=25;namespace=global_ns) #9
	@named gpe1 = GPe(N_inhib=25;namespace=global_ns) #10
	@named gpe2 = GPe(N_inhib=25;namespace=global_ns) #11
	@named STN1 = STN(N_exci=25,I_bg=3*ones(25);namespace=global_ns) #12
    @named STN2 = STN(N_exci=25,I_bg=3*ones(25);namespace=global_ns) #13
	@named Thal1 = Thalamus(N_exci=25;namespace=global_ns) #14
	@named Thal2 = Thalamus(N_exci=25;namespace=global_ns) #15

    fn = joinpath(@__DIR__, "examples/image_example.csv")
    data = CSV.read(fn, DataFrame)
	
    @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=600, t_pause=1000) #16
    @named tan_pop = TAN(;namespace=global_ns) #17

	
	@named AS = GreedyPolicy(namespace=global_ns, t_decision=180.5) #18
    @named SNcb = SNc(namespace=global_ns) #19
	
	
    assembly = [LC, ITN, VC, PFC, STR1, STR2, tan_nrn, gpi1, gpi2, gpe1, gpe2, STN1, STN2, Thal1, Thal2, stim, tan_pop, AS, SNcb]


	hebbian_mod = HebbianModulationPlasticity(K=0.016, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=1600-eps(), t_post=1600-eps(), t_mod=180)
	
    hebbian_cort = HebbianPlasticity(K=5e-5, W_lim=5, t_pre=1600-eps(), t_post=1600-eps())
	
	hebbian_thal_cort = HebbianPlasticity(K=1e-5, W_lim=6, t_pre=1600-eps(), t_post=1600-eps())


	# assembly = [PFC, STR1, STR2]
	
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
	
	
    add_edge!(g,1,3, Dict(:weight => 44)) #LC->VC
	add_edge!(g,1,4, Dict(:weight => 44)) #LC->pfc
	add_edge!(g,2,7, Dict(:weight => 100)) #ITN->tan
	add_edge!(g,3,4, Dict(:weight => 2.5, :density => 0.08, :learning_rule => hebbian_cort)) #VC->pfc
	add_edge!(g,4,5, Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str1
	add_edge!(g,4,6, Dict(:weight => 0.075, :density => 0.04, :learning_rule => hebbian_mod)) #pfc->str2
	add_edge!(g,7,5, Dict(:weight => 0.17)) #tan->str1
	add_edge!(g,7,6, Dict(:weight => 0.17)) #tan->str2
	add_edge!(g,5,8, Dict(:weight => 4, :density => 0.04)) #str1->gpi1
	add_edge!(g,6,9, Dict(:weight => 4, :density => 0.04)) #str2->gpi2
	add_edge!(g,8,14, Dict(:weight => 0.16, :density => 0.04)) #gpi1->thal1
	add_edge!(g,9,15, Dict(:weight => 0.16, :density => 0.04)) #gpi2->thal2
    add_edge!(g,14,4, Dict(:weight => 0.2, :density => 0.32, :learning_rule => hebbian_thal_cort, :sta => true)) #thal1->pfc
	add_edge!(g,15,4, Dict(:weight => 0.2, :density => 0.32, :learning_rule => hebbian_thal_cort, :sta => true)) #thal2->pfc
	add_edge!(g,5,10, Dict(:weight => 4, :density => 0.04)) #str1->gpe1
	add_edge!(g,6,11, Dict(:weight => 4, :density => 0.04)) #str2->gpe2
	add_edge!(g,10,8, Dict(:weight => 0.2, :density => 0.04)) #gpe1->gpi1
	add_edge!(g,11,9, Dict(:weight => 0.2, :density => 0.04)) #gpe2->gpi2
	add_edge!(g,10,12, Dict(:weight => 3.5, :density => 0.04)) #gpe1->stn1
	add_edge!(g,11,13, Dict(:weight => 3.5, :density => 0.04)) #gpe2->stn2
	add_edge!(g,12,8, Dict(:weight => 0.1, :density => 0.04)) #stn1->gpi1
	add_edge!(g,13,9, Dict(:weight => 0.1, :density => 0.04)) #stn2->gpi2
	add_edge!(g,16,3, :weight, 14) #stim->VC
	add_edge!(g,17,5, Dict(:weight => 1, :t_event => 180.0)) #TAN pop -> str1
	add_edge!(g,17,6, Dict(:weight => 1, :t_event => 180.0)) #TAN pop -> str2
	add_edge!(g,5,17, Dict(:weight => 1)) #str1 -> TAN pop 
	add_edge!(g,6,17, Dict(:weight => 1)) #str2 -> TAN pop 
	add_edge!(g,5,6, Dict(:weight => 1, :t_event => 181.0)) #str1 -> str2
	add_edge!(g,6,5, Dict(:weight => 1, :t_event => 181.0)) #str2 -> str1
	add_edge!(g,5,18)# str1->AS
	add_edge!(g,6,18)# str2->AS
	add_edge!(g, 5, 19, Dict(:weight => 1)) # str1->Snc
    add_edge!(g,6, 19, Dict(:weight => 1))  # str2->Snc
	
	
end

# ╔═╡ 48d15d01-7d94-4a2f-bf32-5f0c548b24f6
size(data[1,1:225])

# ╔═╡ 3650464e-90f5-4fd4-abf4-7f0a24e29d41
stim

# ╔═╡ 5ddd418f-23d2-4151-82c0-e8c9fae7e4f2
  agent = Agent(g; name=:ag, t_block = 90);

# ╔═╡ b14520ac-ac20-48e4-ac2a-91448fa398b8
#neuron_net = system_from_graph(g; name=global_ns,t_affect=90.0);

# ╔═╡ 77cbc8f9-5352-447f-b3bd-b81e0f343882
#prob = ODEProblem(structural_simplify(neuron_net), [], (0.0, 2000), []);

# ╔═╡ 7576c26d-8617-4c66-ab5e-db0bd8bb9e17
    prob=agent.problem

# ╔═╡ 58c3e086-8626-4e1d-9a95-7f192ab66666


# ╔═╡ 3db27342-2b9c-4ce3-82d6-cd18e6870dd1
getsys=agent.odesystem;

# ╔═╡ d80b3bf7-24d8-4395-8903-de974e6445f9
begin

	st=states(getsys)
	
    vlist=Int64[]
	
	
	for ii = 1:length(st)
		if contains(string(st[ii]), "V(t)")
			push!(vlist,ii)
		end
	end
end

# ╔═╡ 8049c9d0-328b-4275-8f31-207de4b8f729
size(vlist)

# ╔═╡ 96ca0f9a-fab8-4f9b-b4d2-4dd80dbd8973
sn=(hebbian_mod.modulator)

# ╔═╡ eb6e33a9-3704-46dc-8d51-76c0e6eab779
inh1=collect(0:6:268) .+1

# ╔═╡ 9c276a3f-256c-4cf7-b916-826370d9fffe
inh2=collect(271:6:390) .+1

# ╔═╡ d8445c07-5c91-4170-bd48-8ddbd8de144c
exc1 = setdiff(collect(1:1:270),inh1)

# ╔═╡ 7aab15ca-a27d-436f-b25e-b736d7ad2c3c
exc2 = setdiff(collect(272:1:391),inh2)

# ╔═╡ 49689d91-6efd-4192-b169-5c3b90267d32
str1_ar=collect(392+1:1:392+25)

# ╔═╡ 1e3641bf-497c-426e-b692-b917f4a10ee6
str2_ar=collect(392+26:1:392+50)

# ╔═╡ ad12f058-25bb-4b50-8f35-12544e099242
begin
	thal1_ar=collect(392+50+151:1:392+50+175)
	thal2_ar=collect(392+50+176:1:392+50+200)
end

# ╔═╡ 71ae1e6a-9d52-4481-97e5-a0b911dac68d
size(exc1)

# ╔═╡ c122ffe5-9385-4bf6-83b5-aed97eb74e39


# ╔═╡ 8a774634-557c-4482-9581-623cfd417565
perf=zeros(N_trials)

# ╔═╡ 391c420f-e006-4ad1-8787-6f62b306f33c
act=zeros(N_trials)

# ╔═╡ 709f2677-8627-4069-9827-618b906be7fa
zero([0.9,-1])

# ╔═╡ 78c45058-d2ee-4c1c-92f9-5d43e0647ecf
eps()

# ╔═╡ 0b036e57-120e-4d2f-902e-1352d573bb25
perf

# ╔═╡ a973a05c-e686-48c5-8242-688e6475624b
sum(perf)

# ╔═╡ 432effe7-77c2-4163-99f5-1ba95a3052e8
begin

	 env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    #run_experiment!(agent, env; alg=Vern7(), reltol=1e-9,abstol=1e-9)
end

# ╔═╡ c5bcb85c-0abb-4353-9202-02dcf23125f2
 stim_params2 = Neuroblox.get_trial_stimulus(env)

# ╔═╡ b2b22100-94f5-4d25-93e3-13f387c840f4
   # prob2=remake(prob;tspan=(0,1600))
	prob2 = remake(prob; p = merge(stim_params2),tspan=(0,1600))

# ╔═╡ 0eff1351-d443-4d72-b100-6c591a7c5fe9
    sol = solve(prob2,Vern7(),saveat=0.01)

# ╔═╡ 10cfc03b-525e-4666-be1c-cc5074c40933
    sol.retcode

# ╔═╡ d039f7a0-e15f-4ada-bd0b-3ffac1dd9b96
    ss=convert(Array,sol)

# ╔═╡ 7ca9413b-f672-4177-8289-20b9ceaebc01
aa=findall(x-> x<-59, ss[:,1])

# ╔═╡ 49bd611f-e7f6-4a23-b2b2-13a3edc8bb32
size(aa)

# ╔═╡ 8f9abbff-9755-4da3-ba84-62e4ac7afaa0
aa[end]

# ╔═╡ 6b52f398-7af9-4ce8-bdc4-4943afa09587
ss[:,1]

# ╔═╡ 970dcbaa-2740-4690-933e-7db0d99ebffd
v=ss[vlist[exc2[4]],:]

# ╔═╡ cb8ea960-c0de-4796-8a5f-4f3f357ae774
plot(sol.t,v,xlims=(0,1600))

# ╔═╡ 1f853f06-48ca-4592-8f82-49bdc6e23c68
begin
	VV=zeros(length(vlist),length(sol.t))
	V=zeros(length(vlist),length(sol.t))

	for ii = 1:length(vlist)
		VV[ii,:] .= ss[vlist[ii],:] .+ 200*(ii-1)
		V[ii,:] .= ss[vlist[ii],:] 
	end
end

# ╔═╡ 269c6778-45e5-4e9c-86d6-d01b39bb5d5b
begin
plot(sol.t,[VV[exc1[1:100],:]'],legend=false,yticks=[],color = "blue",size = (1000,700))
plot!(sol.t,[VV[inh1[1:20],:]'],legend=false,yticks=[],color = "red")
#plot!(sol.t,VV[[end],:]',legend=false,yticks=[],color = "green")	
#plot!(soll2.t,[VV[inhib_mod_ar[1],:]],legend=false,yticks=[],color = "green")
	
end

# ╔═╡ 3be065f1-b177-40d6-858b-b43805cf48d3
begin
plot(sol.t,[VV[exc2,:]'],legend=false,yticks=[],color = "blue",size = (1000,700))
plot!(sol.t,[VV[inh2,:]'],legend=false,yticks=[],color = "red")
#plot!(sol.t,VV[[end],:]',legend=false,yticks=[],color = "green")	
#plot!(soll2.t,[VV[inhib_mod_ar[1],:]],legend=false,yticks=[],color = "green")
	
end

# ╔═╡ 8f5bf952-01b1-45d8-9584-5ca31b004d65
begin
plot(sol.t,[VV[str1_ar,:]'],legend=false,yticks=[],color = "green",size = (1000,700));
plot!(sol.t,[VV[str2_ar,:]'],legend=false,yticks=[],color = "blue",size = (1000,700))	
end

# ╔═╡ 327dfad8-ef55-4b43-927b-17448aeb7313
begin
plot(sol.t,[VV[thal1_ar,:]'],legend=false,yticks=[],color = "green",size = (1000,700));
plot!(sol.t,[VV[thal2_ar,:]'],legend=false,yticks=[],color = "blue",size = (1000,700))	
end

# ╔═╡ fc3610bc-8f62-4f23-b740-527eb2a94e79
plot(sol.t,ss[vlist[exc2[4]]+8,:])

# ╔═╡ 7f6120ca-61bc-4161-b827-26560bc5f28c
plot(sol.t,mean(V[exc1,:],dims=1)')

# ╔═╡ fe451072-9911-41d6-b179-aad13f38dce0
    action = agent.action_selection(agent.odesystem,prob2)

# ╔═╡ 14e2806e-5cdd-4b0b-b8c7-8c6d2c7e349c
begin 

	#function run_experiment_test!(agent::Agent, env::ClassificationEnvironment, t_warmup=200.0; kwargs...)
	t_warmup=800
    #N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)

    sys = Neuroblox.get_sys(agent)
    prob3 = remake(prob; p = merge(stim_params2),tspan=(0,1600))

	if t_warmup > 0
        prob3 = remake(prob3; tspan=(0,t_warmup))
        #if haskey(kwargs, :alg)
        #    sol = solve(prob, kwargs[:alg]; kwargs...)
        #else
		
            sol2 = solve(prob3, Vern7())
		
        #end
        u0 = sol2[1:end,end] # last value of state vector
        prob3 = remake(prob3; p = merge(stim_params2),tspan=tspan, u0=u0)
    else
        prob3 = remake(prob3; p = merge(stim_params2),tspan=tspan)
        u0 = []
    end

    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    
    defs = ModelingToolkit.get_defaults(sys)
    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

end


# ╔═╡ dedebc43-46f3-47d1-aaeb-da74b7b5c588
size(u0)

# ╔═╡ 6aabdb11-ff58-4434-bdff-5c2d7e80ed68
weights

# ╔═╡ b7e84b20-0b80-478c-bc88-2883f80bcbb4
begin
learning=1
	
	if learning==1
		
    for ii = 1:N_trials
        prob4 = agent.problem
        stim_params = Neuroblox.get_trial_stimulus(env)
        prob4 = remake(prob4; p = merge(weights, stim_params), u0 = u0,tspan=(0,1600))

        #if haskey(kwargs, :alg)
          #  sol = solve(prob, kwargs[:alg]; kwargs...)
        #else
            sol2 = solve(prob4, Vern7())
        #end
		    agent.problem = prob4
            @info env.current_trial
        #u0 = sol[1:end,end] # next run should continue where the last one ended

        if isnothing(action_selection)
            feedback = 1
        else
           # action = action_selection(sol2)
			action = action_selection(agent)
            feedback = env(action)
			@info action
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
	
end

# ╔═╡ 650ed1b7-1643-4e74-8a61-f1fbd70ba229
env.current_trial

# ╔═╡ Cell order:
# ╠═7d500a22-c5ed-48d1-b009-4d4a0f03fc2a
# ╠═98f74502-74e6-11ee-34ff-7b16a35b2860
# ╠═b6a6209f-97f3-4844-928a-948f3d53b4d7
# ╠═1f184999-9dd8-4958-8158-157b683b6f5c
# ╠═536b8983-8548-4144-be00-4d516fe7b75f
# ╠═e6a453f3-e2a8-44ec-9be2-1f6d9293c911
# ╠═038809e1-7512-4f5d-8fbf-feef2cdce117
# ╠═e8deef43-7ee9-484f-a8a8-2df6e7006e22
# ╠═b4e8ba6e-e91a-474c-9314-8dd75fa2e82e
# ╠═ee0c04b9-57c9-44bc-b092-218019a545ec
# ╠═d37ba3dd-78df-4e18-ac76-3864a2e9216e
# ╠═8abc4a4b-4acf-4c25-a10c-33278329c9f4
# ╠═48d15d01-7d94-4a2f-bf32-5f0c548b24f6
# ╠═3650464e-90f5-4fd4-abf4-7f0a24e29d41
# ╠═5ddd418f-23d2-4151-82c0-e8c9fae7e4f2
# ╠═b14520ac-ac20-48e4-ac2a-91448fa398b8
# ╠═77cbc8f9-5352-447f-b3bd-b81e0f343882
# ╠═7576c26d-8617-4c66-ab5e-db0bd8bb9e17
# ╠═58c3e086-8626-4e1d-9a95-7f192ab66666
# ╠═3db27342-2b9c-4ce3-82d6-cd18e6870dd1
# ╠═d80b3bf7-24d8-4395-8903-de974e6445f9
# ╠═8049c9d0-328b-4275-8f31-207de4b8f729
# ╠═b2b22100-94f5-4d25-93e3-13f387c840f4
# ╠═0eff1351-d443-4d72-b100-6c591a7c5fe9
# ╠═fe451072-9911-41d6-b179-aad13f38dce0
# ╠═96ca0f9a-fab8-4f9b-b4d2-4dd80dbd8973
# ╠═10cfc03b-525e-4666-be1c-cc5074c40933
# ╠═d039f7a0-e15f-4ada-bd0b-3ffac1dd9b96
# ╠═7ca9413b-f672-4177-8289-20b9ceaebc01
# ╠═49bd611f-e7f6-4a23-b2b2-13a3edc8bb32
# ╠═6b52f398-7af9-4ce8-bdc4-4943afa09587
# ╠═970dcbaa-2740-4690-933e-7db0d99ebffd
# ╠═cb8ea960-c0de-4796-8a5f-4f3f357ae774
# ╠═eb6e33a9-3704-46dc-8d51-76c0e6eab779
# ╠═9c276a3f-256c-4cf7-b916-826370d9fffe
# ╠═d8445c07-5c91-4170-bd48-8ddbd8de144c
# ╠═7aab15ca-a27d-436f-b25e-b736d7ad2c3c
# ╠═49689d91-6efd-4192-b169-5c3b90267d32
# ╠═1e3641bf-497c-426e-b692-b917f4a10ee6
# ╠═ad12f058-25bb-4b50-8f35-12544e099242
# ╠═71ae1e6a-9d52-4481-97e5-a0b911dac68d
# ╠═1f853f06-48ca-4592-8f82-49bdc6e23c68
# ╠═269c6778-45e5-4e9c-86d6-d01b39bb5d5b
# ╠═3be065f1-b177-40d6-858b-b43805cf48d3
# ╠═8f5bf952-01b1-45d8-9584-5ca31b004d65
# ╠═327dfad8-ef55-4b43-927b-17448aeb7313
# ╠═c122ffe5-9385-4bf6-83b5-aed97eb74e39
# ╠═fc3610bc-8f62-4f23-b740-527eb2a94e79
# ╠═8f9abbff-9755-4da3-ba84-62e4ac7afaa0
# ╠═7f6120ca-61bc-4161-b827-26560bc5f28c
# ╠═dedebc43-46f3-47d1-aaeb-da74b7b5c588
# ╠═14e2806e-5cdd-4b0b-b8c7-8c6d2c7e349c
# ╠═8a774634-557c-4482-9581-623cfd417565
# ╠═391c420f-e006-4ad1-8787-6f62b306f33c
# ╠═c5bcb85c-0abb-4353-9202-02dcf23125f2
# ╠═6aabdb11-ff58-4434-bdff-5c2d7e80ed68
# ╠═709f2677-8627-4069-9827-618b906be7fa
# ╠═78c45058-d2ee-4c1c-92f9-5d43e0647ecf
# ╠═b7e84b20-0b80-478c-bc88-2883f80bcbb4
# ╠═650ed1b7-1643-4e74-8a61-f1fbd70ba229
# ╠═0b036e57-120e-4d2f-902e-1352d573bb25
# ╠═a973a05c-e686-48c5-8242-688e6475624b
# ╠═432effe7-77c2-4163-99f5-1ba95a3052e8
