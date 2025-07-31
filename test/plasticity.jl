using Neuroblox
using OrdinaryDiffEq
using Test
using Graphs
using MetaGraphs
using DataFrames
using CSV
using ModelingToolkit: getp

@testset "Cortical-Cortical plasticity test" begin
    t_trial = 100 # ms
    time_block_dur = 10 # ms
    N_trials = 3

    global_ns = :g # global namespace
    @named VAC = CorticalBlox(N_wta=3, N_exci=3, namespace=global_ns, density=0.1, weight=1)
    @named PFC = CorticalBlox(N_wta=2, N_exci=3, namespace=global_ns, density=0.1, weight=1)

    fn = joinpath(@__DIR__, "../examples/image_example.csv")
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
    ps = parameters(agent.system)
    init_params = agent.problem.p
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    idxs_weight = findall(x -> occursin("w_", String(Symbol(x))), ps)
    idx_stim = findall(x -> occursin("stim₊", String(Symbol(x))), ps)
    idx_jcn = findall(x -> occursin("jcn", String(Symbol(x))), ps)
    idxs_other_params = setdiff(eachindex(ps), vcat(idxs_weight, idx_stim, idx_jcn))


    params_at(idxs) = getp(agent.problem, parameters(agent.system)[idxs])(agent.problem)
    init_params_all = params_at(:)
    init_params_idxs_weight = params_at(idxs_weight)
    init_params_idxs_other_params = params_at(idxs_other_params)

    
    env = ClassificationEnvironment(stim; name=:env, namespace=global_ns)
    run_experiment!(agent, env; t_warmup=200, alg=Tsit5(), reltol=1e-6,abstol=1e-9)

    final_params = agent.problem.p
    # At least some weights need to be different.
    @test any(init_params_idxs_weight .!= params_at(idxs_weight))
    # All non-weight parameters need to be the same.
    @test all(init_params_idxs_other_params .== params_at(idxs_other_params))
end
