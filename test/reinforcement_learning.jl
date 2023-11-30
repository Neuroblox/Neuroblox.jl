using Neuroblox
using DifferentialEquations
using Test
using Graphs
using MetaGraphs
using DataFrames
using CSV

@testset "RFLearning with policy" begin
    t_trial = 2 # ms
    time_block_dur = 0.01 # ms
    N_trials = 3

    global_ns = :g # global namespace
    @named VAC = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named PFC = CorticalBlox(N_wta=2, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named STR_L = Striatum(N_inhib=2, namespace=global_ns)
    @named STR_R = Striatum(N_inhib=2, namespace=global_ns)
    @named SNcb = SNc(namespace=global_ns, N_time_blocks=t_trial/time_block_dur)

    @named AS = GreedyPolicy(namespace=global_ns, t_decision=0.31*t_trial)

    fn = "../examples/image_example.csv"
    data = CSV.read(fn, DataFrame)
    @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=0.4*t_trial, t_pause=0.6*t_trial)

    bloxs = [VAC, PFC, STR_L, STR_R, SNcb, AS, stim]
    d = Dict(b => i for (i,b) in enumerate(bloxs))

    hebbian_mod = HebbianModulationPlasticity(K=0.2, decay=0.01, modulator=SNcb, t_pre=t_trial, t_post=t_trial, t_mod=0.31*t_trial)
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

    agent = Agent(g; name=:ag, t_block = t_trial/5);
    ps = parameters(agent.odesystem)
    init_params = agent.problem.p
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    idxs_weight = findall(x -> occursin("w_", String(Symbol(x))), ps)
    idx_stim = findall(x -> occursin("stim₊", String(Symbol(x))), ps)
    idx_jcn = findall(x -> occursin("jcn", String(Symbol(x))), ps)
    idxs_other_params = setdiff(eachindex(ps), vcat(idxs_weight, idx_stim, idx_jcn))

    env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    run_experiment!(agent, env; alg=QNDF(), reltol=1e-9,abstol=1e-9)

    final_params = agent.problem.p
    # At least some weights need to be different.
    @test any(init_params[map_idxs[idxs_weight]] .!= final_params[map_idxs[idxs_weight]])
    # All non-weight parameters need to be the same.
    @test all(init_params[map_idxs[idxs_other_params]] .== final_params[map_idxs[idxs_other_params]])
end

@testset "RFLearning without policy" begin
    t_trial = 100 # ms
    time_block_dur = 10 # ms
    N_trials = 3

    global_ns = :g # global namespace
    @named VAC = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named PFC = CorticalBlox(N_wta=2, N_exci=3, namespace=global_ns, density=0.1, weight=1)

    fn = "../examples/image_example.csv"
    data = CSV.read(fn, DataFrame)
    @named stim = ImageStimulus(data[1:N_trials,:]; namespace=global_ns, t_stimulus=0.4*t_trial, t_pause=0.6*t_trial)

    bloxs = [VAC, PFC, stim]

    d = Dict(b => i for (i,b) in enumerate(bloxs))

    hebbian = HebbianPlasticity(K=0.2, W_lim=2, t_pre=t_trial, t_post=t_trial)

    g = MetaDiGraph()
    add_blox!.(Ref(g), bloxs)

    add_edge!(g, d[stim], d[VAC], Dict(:weight => 15, :density => 0.1))
    add_edge!(g, d[VAC], d[PFC], Dict(:weight => 1, :density => 0.1, :learning_rule => hebbian))

    agent = Agent(g; name=:ag);
    ps = parameters(agent.odesystem)
    init_params = agent.problem.p
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    idxs_weight = findall(x -> occursin("w_", String(Symbol(x))), ps)
    idx_stim = findall(x -> occursin("stim₊", String(Symbol(x))), ps)
    idx_jcn = findall(x -> occursin("jcn", String(Symbol(x))), ps)
    idxs_other_params = setdiff(eachindex(ps), vcat(idxs_weight, idx_stim, idx_jcn))

    env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    run_experiment_open_loop!(agent, env; alg=Tsit5(), reltol=1e-6,abstol=1e-9)

    final_params = agent.problem.p
    # At least some weights need to be different.
    @test any(init_params[map_idxs[idxs_weight]] .!= final_params[map_idxs[idxs_weight]])
    # All non-weight parameters need to be the same.
    @test all(init_params[map_idxs[idxs_other_params]] .== final_params[map_idxs[idxs_other_params]])
end
