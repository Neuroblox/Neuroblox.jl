using Neuroblox
using OrdinaryDiffEq
using Test
using DataFrames
using CSV
using ModelingToolkit: getp

@testset "RL test" begin
    t_trial = 2 # ms
    time_block_dur = 0.01 # ms
    N_trials = 3

    global_ns = :g # global namespace
    @named VAC = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named PFC = CorticalBlox(N_wta=2, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named STR_L = Striatum(N_inhib=2, namespace=global_ns)
    @named STR_R = Striatum(N_inhib=2, namespace=global_ns)
    @named SNcb = SNc(namespace=global_ns, N_time_blocks=t_trial/time_block_dur)
    @named TAN_pop = TAN(;namespace=global_ns)

    @named AS = GreedyPolicy(namespace=global_ns, t_decision=0.31*t_trial)

    fn = joinpath(@__DIR__, "../examples/image_example.csv")
    @named stim = ImageStimulus(fn; namespace=global_ns, t_stimulus=0.4*t_trial, t_pause=0.6*t_trial)

    bloxs = [VAC, PFC, STR_L, STR_R, SNcb, TAN_pop, AS, stim]
    d = Dict(b => i for (i,b) in enumerate(bloxs))

    hebbian_mod = HebbianModulationPlasticity(K=0.2, decay=0.01, α=3, θₘ=1, modulator=SNcb, t_pre=t_trial, t_post=t_trial, t_mod=0.31*t_trial)
    hebbian = HebbianPlasticity(K=0.2, W_lim=2, t_pre=t_trial, t_post=t_trial)

    g = MetaDiGraph()
    add_blox!.(Ref(g), bloxs)

    add_edge!(g, d[stim], d[VAC], Dict(:weight => 1, :density => 0.1))
    add_edge!(g, d[VAC], d[PFC], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian))
    add_edge!(g, d[PFC], d[STR_L], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian_mod))
    add_edge!(g, d[PFC], d[STR_R], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian_mod))
    add_edge!(g, d[STR_R], d[STR_L], Dict(:weight => 1, :t_event => 0.3*t_trial))
    add_edge!(g, d[STR_L], d[STR_R], Dict(:weight => 1, :t_event => 0.3*t_trial))
    add_edge!(g, d[STR_L], d[SNcb], Dict(:weight => 1))
    add_edge!(g, d[STR_R], d[SNcb], Dict(:weight => 1))
    add_edge!(g, d[STR_L], d[AS])
    add_edge!(g, d[STR_R], d[AS])
    add_edge!(g, d[STR_L], d[TAN_pop], Dict(:weight => 1))
    add_edge!(g, d[STR_R], d[TAN_pop], Dict(:weight => 1))
    add_edge!(g, d[TAN_pop], d[STR_L], Dict(:weight => 1, :t_event => 0.1*t_trial))
    add_edge!(g, d[TAN_pop], d[STR_R], Dict(:weight => 1, :t_event => 0.1*t_trial))

    agent = Agent(g; name=:ag, t_block = t_trial/5);
    ps = parameters(agent.system)

    
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    idxs_weight = findall(x -> occursin("w_", String(Symbol(x))), ps)
    idx_stim = findall(x -> occursin("stim₊", String(Symbol(x))), ps)
    idx_jcn = findall(x -> occursin("jcn", String(Symbol(x))), ps)
    idx_spikes = findall(x -> occursin("spikes", String(Symbol(x))), ps)
    idx_H = findall(x -> occursin("H", String(Symbol(x))), ps)
    idx_I_bg = findall(x -> occursin("I_bg", String(Symbol(x))), ps)
    idxs_other_params = setdiff(eachindex(ps), vcat(idxs_weight, idx_stim, idx_jcn, idx_spikes, idx_H, idx_I_bg))

    params_at(idxs) = getp(agent.problem, parameters(agent.system)[idxs])(agent.problem)
    init_params_all = params_at(:)
    init_params_idxs_weight = params_at(idxs_weight)
    init_params_idxs_other_params = params_at(idxs_other_params)
    
    env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    run_experiment!(agent, env; t_warmup=200, alg=Vern7(), reltol=1e-9,abstol=1e-9)
    
    final_params = reduce(vcat, agent.problem.p)
    # At least some weights need to be different.
    @test any(init_params_idxs_weight .!= params_at(idxs_weight))
    # @test any(init_params[map_idxs[idxs_weight]] .!= final_params[map_idxs[idxs_weight]])
    # All non-weight parameters need to be the same.
    @test all(init_params_idxs_other_params .== params_at(idxs_other_params))
    # @test all(init_params[map_idxs[idxs_other_params]] .== final_params[map_idxs[idxs_other_params]])

    reset!(env)
    @test env.current_trial == 1
end

@testset "RL test with save" begin
    t_trial = 2 # ms
    time_block_dur = 0.01 # ms
    N_trials = 3

    global_ns = :g # global namespace
    @named VAC = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named PFC = CorticalBlox(N_wta=2, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named STR_L = Striatum(N_inhib=2, namespace=global_ns)
    @named STR_R = Striatum(N_inhib=2, namespace=global_ns)
    @named SNcb = SNc(namespace=global_ns, N_time_blocks=t_trial/time_block_dur)
    @named TAN_pop = TAN(;namespace=global_ns)

    @named AS = GreedyPolicy(namespace=global_ns, t_decision=0.31*t_trial)

    fn = joinpath(@__DIR__, "../examples/image_example.csv")
    data = CSV.read(fn, DataFrame)
    @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=0.4*t_trial, t_pause=0.6*t_trial)

    bloxs = [VAC, PFC, STR_L, STR_R, SNcb, TAN_pop, AS, stim]
    d = Dict(b => i for (i,b) in enumerate(bloxs))

    hebbian_mod = HebbianModulationPlasticity(K=0.2, decay=0.01, α=3, θₘ=1, modulator=SNcb, t_pre=t_trial, t_post=t_trial, t_mod=0.31*t_trial)
    hebbian = HebbianPlasticity(K=0.2, W_lim=2, t_pre=t_trial, t_post=t_trial)

    g = MetaDiGraph()
    add_blox!.(Ref(g), bloxs)

    add_edge!(g, d[stim], d[VAC], Dict(:weight => 1, :density => 0.1))
    add_edge!(g, d[VAC], d[PFC], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian))
    add_edge!(g, d[PFC], d[STR_L], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian_mod))
    add_edge!(g, d[PFC], d[STR_R], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian_mod))
    add_edge!(g, d[STR_R], d[STR_L], Dict(:weight => 1, :t_event => 0.3*t_trial))
    add_edge!(g, d[STR_L], d[STR_R], Dict(:weight => 1, :t_event => 0.3*t_trial))
    add_edge!(g, d[STR_L], d[SNcb], Dict(:weight => 1))
    add_edge!(g, d[STR_R], d[SNcb], Dict(:weight => 1))
    add_edge!(g, d[STR_L], d[AS])
    add_edge!(g, d[STR_R], d[AS])
    add_edge!(g, d[STR_L], d[TAN_pop], Dict(:weight => 1))
    add_edge!(g, d[STR_R], d[TAN_pop], Dict(:weight => 1))
    add_edge!(g, d[TAN_pop], d[STR_L], Dict(:weight => 1, :t_event => 0.1*t_trial))
    add_edge!(g, d[TAN_pop], d[STR_R], Dict(:weight => 1, :t_event => 0.1*t_trial))

    agent = Agent(g; name=:ag, t_block = t_trial/5);
    ps = parameters(agent.system)

    
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    idxs_weight = findall(x -> occursin("w_", String(Symbol(x))), ps)
    idx_stim = findall(x -> occursin("stim₊", String(Symbol(x))), ps)
    idx_jcn = findall(x -> occursin("jcn", String(Symbol(x))), ps)
    idx_spikes = findall(x -> occursin("spikes", String(Symbol(x))), ps)
    idx_H = findall(x -> occursin("H", String(Symbol(x))), ps)
    idx_I_bg = findall(x -> occursin("I_bg", String(Symbol(x))), ps)
    idxs_other_params = setdiff(eachindex(ps), vcat(idxs_weight, idx_stim, idx_jcn, idx_spikes, idx_H, idx_I_bg))

    params_at(idxs) = getp(agent.problem, parameters(agent.system)[idxs])(agent.problem)
    init_params_all = params_at(:)
    init_params_idxs_weight = params_at(idxs_weight)
    init_params_idxs_other_params = params_at(idxs_other_params)
        
    env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    run_experiment!(agent, env, "./"; t_warmup=200, alg=Vern7(), reltol=1e-9,abstol=1e-9)
    
    final_params = reduce(vcat, agent.problem.p)
    # At least some weights need to be different.
    @test any(init_params_idxs_weight .!= params_at(idxs_weight))
    # @test any(init_params[map_idxs[idxs_weight]] .!= final_params[map_idxs[idxs_weight]])
    # All non-weight parameters need to be the same.
    @test all(init_params_idxs_other_params .== params_at(idxs_other_params))
    # @test all(init_params[map_idxs[idxs_other_params]] .== final_params[map_idxs[idxs_other_params]])

    reset!(env)
    @test env.current_trial == 1
end