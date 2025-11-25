"""
    HebbianPlasticity(; K, W_lim,
                      state_pre = nothing,
                      state_post = nothing,
                      t_pre = nothing,
                      t_post = nothing)

Hebbian learning rule. Every trial of the RL experiment, update the weight according to the following:

```math
    w_{j+1} = w_j + \\text{feedback} × Kx_\\text{pre}x_\\text{post}(W_\\text{lim} - w)
```
where `feedback` indicates the correctness of the agent's action during the trial, and the `x` indicate the activities of the pre- and post-synaptic neurons.

Arguments:
    - K: the learning rate of the connection
    - W_lim: the maximum weight for the connection

See also [`HebbianModulationPlasticity`](@ref).
"""
mutable struct HebbianPlasticity <:AbstractLearningRule
    const K::Float64
    const W_lim::Float64
    state_pre::Union{Nothing, Num, Symbol}
    state_post::Union{Nothing, Num, Symbol}
    t_pre::Union{Nothing, Float64}
    t_post::Union{Nothing, Float64}
end
function HebbianPlasticity(;
        K, W_lim, 
        state_pre=nothing, state_post=nothing,
        t_pre=nothing, t_post=nothing
    )
    HebbianPlasticity(K, W_lim, state_pre, state_post, t_pre, t_post)
end

function (hp::HebbianPlasticity)(val_pre, val_post, w, feedback)
    Δw = hp.K * val_pre * val_post * (hp.W_lim - w) * feedback

    return Δw
end

function weight_gradient(hp::HebbianPlasticity, sol, w, feedback)
    val_pre = only(sol(hp.t_pre; idxs = [hp.state_pre]))
    val_post = only(sol(hp.t_post; idxs = [hp.state_post]))

    return hp(val_pre, val_post, w, feedback)
end

get_eval_times(l::HebbianPlasticity) = [l.t_pre, l.t_post]

get_eval_states(l::HebbianPlasticity) = [l.state_pre, l.state_post]

"""
    HebbianModulationPlasticity(; K, decay, α, θₘ,
                                state_pre = nothing,
                                state_post = nothing,
                                t_pre = nothing,
                                t_post = nothing,
                                t_mod = nothing,
                                modulator = nothing)

Hebbian learning rule, but modulated by the dopamine reward prediction error. The weight update is largest when the reward prediction error is far from the modulation threshold θₘ.

```math
    ϵ = \\text{feedback} - (\\text{DA}_b - \\text{DA})
    w_{j+1} = w_j + \\max(\\times Kx_\\text{pre}x_\\text{post}ϵ(ϵ + θₘ) dσ(α(ϵ + θₘ)) - \\text{decay} × w, -w)
```
where `feedback` indicates the correctness of the agent's action during the trial, DA_b is the baseline dopamine level and DA is the modulator's dopamine release, and dσ is the derivative of the logistic function. The decay prevents the weights from diverging.

Arguments:
    - K: the learning rate of the connection
    - decay: Decay of the weight update
    - α: the selectivity of the derivative of the logistic function
    - θₘ: the modulation threshold for the reward prediction error

See also [`HebbianPlasticity`](@ref).
"""
mutable struct HebbianModulationPlasticity <: AbstractLearningRule
    const K::Float64
    const decay::Float64
    const α::Float64
    const θₘ::Float64
    state_pre::Union{Nothing, Num, Symbol}
    state_post::Union{Nothing, Num, Symbol}
    t_pre::Union{Nothing, Float64}
    t_post::Union{Nothing, Float64}
    t_mod::Union{Nothing, Float64}
    modulator
end
function HebbianModulationPlasticity(; 
    K, decay, α, θₘ, modulator=nothing,
    state_pre=nothing, state_post=nothing, 
    t_pre=nothing, t_post=nothing, t_mod=nothing,   
)
    HebbianModulationPlasticity(K, decay, α, θₘ, state_pre, state_post, t_pre, t_post, t_mod, modulator)
end

dlogistic(x) = logistic(x) * (1 - logistic(x)) 

function (hmp::HebbianModulationPlasticity)(val_pre, val_post, val_modulator, w, feedback)
    DA = hmp.modulator(val_modulator)
    DA_baseline = hmp.modulator.κ_DA * hmp.modulator.N_time_blocks
    ϵ = feedback - (hmp.modulator.κ_DA - DA)
    
   # Δw = hmp.K * val_post * val_pre * DA * (DA - DA_baseline) * dlogistic(DA) - hmp.decay * w
    Δw = max((hmp.K * val_post * val_pre * ϵ * (ϵ + hmp.θₘ) * dlogistic(hmp.α * (ϵ + hmp.θₘ)) - hmp.decay * w), -w)

    return Δw
end

function weight_gradient(hmp::HebbianModulationPlasticity, sol, w, feedback)
    state_mod = get_modulator_state(hmp.modulator)
    val_pre = sol(hmp.t_pre; idxs = hmp.state_pre)
    val_post = sol(hmp.t_post; idxs = hmp.state_post)
    val_mod = sol(hmp.t_mod; idxs = state_mod)

    return hmp(val_pre, val_post, val_mod, w, feedback)
end

get_eval_times(l::HebbianModulationPlasticity) = [l.t_pre, l.t_post, l.t_mod]

get_eval_states(l::HebbianModulationPlasticity) = [l.state_pre, l.state_post, get_modulator_state(l.modulator)]

"""
    ClassificationEnvironment(stim::ImageStimulus, N_trials; name, namespace, t_stimulus, t_pause)

Create an environment for reinforcement learning. A set of images is presented to the agent to be classified. This struct stores the correct class for each image, and the current trial of the experiment. 

Arguments:
- stim: The ImageStimulus, created from a set of images
- N_trials: Number of trials. The agent performs one classification each trial.
- t_stimulus: The length of time the stimulus is on (ms)
- t_pause: The length of time the stimulus is off (ms)
- 
"""
mutable struct ClassificationEnvironment{S} <: AbstractEnvironment
    const name::Symbol
    const namespace::Symbol
    const source::S
    const category::Vector{Int}
    const N_trials::Int
    const t_trial::Float64
    current_trial::Int
    
    function ClassificationEnvironment(data::DataFrame; name, namespace=nothing, t_stimulus, t_pause)
        stim = ImageStimulus(
                        data; 
                        name=:stim, 
                        namespace=namespaced_name(namespace, name),
                        t_stimulus,
                        t_pause
        )
        
        N_trials = stim.N_stimuli

        ClassificationEnvironment(stim, N_trials; name, namespace)
    end

    function ClassificationEnvironment(data::DataFrame, N_trials; name, namespace=nothing, t_stimulus, t_pause)
        stim = ImageStimulus(
                        data; 
                        name=:stim, 
                        namespace=namespaced_name(namespace, name),
                        t_stimulus,
                        t_pause
        )

        ClassificationEnvironment(stim, N_trials; name, namespace)
    end
    
    function ClassificationEnvironment(stim::ImageStimulus; name, namespace=nothing)
        N_trials = stim.N_stimuli

        ClassificationEnvironment(stim, N_trials; name, namespace)
    end

    function ClassificationEnvironment(stim::ImageStimulus, N_trials; name, namespace=nothing)
        t_trial = stim.t_stimulus + stim.t_pause

        new{typeof(stim)}(Symbol(name), Symbol(namespace), stim, stim.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]

function get_trial_stimulus(env::ClassificationEnvironment)
    stim_params = env.source.stim_parameters
    stim_values = env.source.IMG[:, env.current_trial]

    return Dict(p => v for (p, v) in zip(stim_params, stim_values))
end

"""
    GreedyPolicy(; name, t_decision, namespace, competitor_states = Num[], competitor_params = Num[])

A policy that performs classification by picking the state with the highest value among `competitor_states`. `t_decision` is the time of the decision.
"""
mutable struct GreedyPolicy <: AbstractActionSelection
    const name::Symbol
    const namespace::Symbol
    competitor_states::Union{Vector{Symbol}, Vector{Num}}
    competitor_params::Union{Vector{Symbol}, Vector{Num}}
    const t_decision::Float64

    function GreedyPolicy(; name, t_decision, namespace=nothing, competitor_states=Num[], competitor_params=Num[])
        new(name, namespace, competitor_states, competitor_params, t_decision)
    end
end

function (p::GreedyPolicy)(sol::AbstractSolution)
    comp_vals = map(p.competitor_states) do sym
        sol(p.t_decision; idxs=sym)
    end
    return argmax(comp_vals)
end

get_eval_times(gp::GreedyPolicy) = [gp.t_decision]

get_eval_states(gp::GreedyPolicy) = gp.competitor_states

mutable struct MTKAgent{S,P,A,LR,C} <: AbstractAgent
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connector::C
end

"""
    Agent(g::MetaDiGraph; name, graphdynamics=false, kwargs...)

Create a RL agent from a graph representing a neural circuit. This contains the system constructed from the graph, as well as its policy, connections, and the learning rules of each connection, which are extracted from the graph. The `graphdynamics` kwarg sets whether to construct a GraphDynamics system or ModelingToolkit system from the graph.
"""
function Agent(g::MetaDiGraph; name, graphdynamics=false, kwargs...)
    if graphdynamics
        gsys = to_graphsystem(g)
        return Agent(gsys; name, kwargs...)
    end
    conns = connectors_from_graph(g)
    
    t_block = haskey(kwargs, :t_block) ? kwargs[:t_block] : missing
    # TODO: add another version that uses system_from_graph(g,bc,params;)
    sys = system_from_graph(g, conns; name, t_block, allow_parameter=false)

    u0 = haskey(kwargs, :u0) ? kwargs[:u0] : []
    p = haskey(kwargs, :p) ? kwargs[:p] : []
    
    prob = ODEProblem(sys, u0, (0.,1.), p)
    
    policy = action_selection_from_graph(g)
    lr =  narrowtype(learning_rules(conns))  

    MTKAgent(sys, prob, policy, lr, conns)
end

function run_experiment!(agent::MTKAgent,
                         env::ClassificationEnvironment,
                         save_path=nothing;
                         monitor=nothing,
                         modulator=nothing,
                         t_warmup=0,
                         interrupt_token = Threads.Atomic{Bool}(false),
                         kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)

    sys = get_system(agent)
    defs = ModelingToolkit.get_defaults(sys)
    learning_rules = agent.learning_rules

    stim_params = get_trial_stimulus(env)
    init_params = ModelingToolkit.MTKParameters(sys, merge(defs, stim_params))

    if t_warmup > 0
        u0 = run_warmup(agent, env, t_warmup; kwargs...)
        agent.problem = remake(agent.problem; tspan, u0=u0, p=init_params)
    else
        agent.problem = remake(agent.problem; tspan, p=init_params)
    end
    
    

    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

    #=
    # TO DO: Ideally we should use save_idxs here to save some memory for long solves.
    # However it does not seem possible currently to either do time interpolation on the solution
    # or access observed states when save_idxs is used. Need to check with SciML people.
    states = unknowns(sys)
    idxs_V = findall(s -> occursin("₊V(t)", s), String.(Symbol.(states)))

    states_learning = mapreduce(get_eval_states, union, values(learning_rules))
    action_selection = agent.action_selection 
    if !isnothing(action_selection)
        states_learning = union(states_learning, get_eval_states(action_selection))
    end
    
    idxs_learning = map(states_learning) do sl
        findfirst(s -> occursin(String(Symbol(sl)), String(Symbol(s))), states)
    end
    filter!(!isnothing, idxs_learning)
    
    save_idxs = union(idxs_V, idxs_learning)
    =#
    if isnothing(modulator)
        trace = (; trial=Int[], correct=Bool[], action=Int[], time=Float64[])
    else
        trace = (; trial=Int[], correct=Bool[], action=Int[], time=Float64[], DA=Float64[])
    end
    try 
        for trial in 1:N_trials
            if interrupt_token[]
                @warn "Interrupted! Bailing out now"
                break
            end
            (;time,) = @timed begin
                sol, iscorrect, action = run_trial!(agent, env, weights, nothing; kwargs...)
            end
            if !isnothing(save_path)
                save_voltages(sol, save_path, trial)
            end
            push!(trace.trial, trial)
            push!(trace.correct, iscorrect)
            push!(trace.action, action)
            push!(trace.time, time)
            if !isnothing(modulator)
                push!(trace.DA, get_DA(sol, modulator, iscorrect))
            end
            update_monitor!(monitor, sol, action, iscorrect, trace)
        end
    catch e;
        if e isa InterruptException
            @warn "Interrupted! Bailing out now"
        else
            rethrow(e)
        end
    end
    return trace
end

function run_warmup(agent::MTKAgent, env::ClassificationEnvironment, t_warmup; kwargs...)

    prob = remake(agent.problem; tspan=(0, t_warmup))
    if haskey(kwargs, :alg)
        sol = solve(prob, kwargs[:alg]; save_everystep=false, kwargs...)
    else
        sol = solve(prob; alg_hints = [:stiff], save_everystep=false, kwargs...)
    end
    u0 = sol[:,end] # last value of state vector

    return u0
end

function run_trial!(agent::MTKAgent, env::ClassificationEnvironment, weights, u0; kwargs...)

    prob = agent.problem
    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    sys = get_system(agent)
    defs = ModelingToolkit.get_defaults(sys)

    if haskey(kwargs, :alg)
        sol = solve(prob, kwargs[:alg]; kwargs...)
    else
        sol = solve(prob; alg_hints = [:stiff], kwargs...)
    end

    # u0 = sol[1:end,end] # next run should continue where the last one ended   
    # In the paper we assume sufficient time interval before next stimulus so that
    # system reaches back to steady state, so we don't continue from previous trial's endpoint

    if isnothing(action_selection)
        feedback = 1
        action = 0
    else
        action = action_selection(sol)
        feedback = env(action)
    end

    for (w, rule) in learning_rules
        w_val = weights[w]
        Δw = weight_gradient(rule, sol, w_val, feedback)
        weights[w] += Δw
    end
    
    increment_trial!(env)

    stim_params = get_trial_stimulus(env)
    new_params = ModelingToolkit.MTKParameters(sys, merge(defs, weights, stim_params))

    agent.problem = remake(prob; p = new_params)

    return sol, feedback, action
end


function save_voltages(sol, filepath, numtrial)
    df = DataFrame(sol)
    fname = "sim"*lpad(numtrial, 4, "0")*".csv"
    fullpath = joinpath(filepath, fname)
    write(fullpath, df)
end

function save_voltages_block(sol, filepath, numtrial, blocks)
    df2 = DataFrame()
    df2.t = sol.t
    
    # Iterate through all blocks and add their meanfield timeseries
    for (name, block) in pairs(blocks)
        df2[!, Symbol(name)] = meanfield_timeseries(block, sol)
    end
    
    fname = "$(numtrial)_meanfield.csv"
    fullpath = joinpath(filepath, fname)
    CSV.write(fullpath, df2, writeheader=true)
end

function get_val_modulator(sol, modulator)
    #change this harcoded value, hacky for now
    t_mod = 90
    state_mod = get_modulator_state(modulator)
    return only(sol(t_mod; idxs = state_mod))
end

function get_DA(sol, modulator, feedback)
    val_modulator = get_val_modulator(sol, modulator)
    DA = modulator(val_modulator)
    ϵ = feedback - (modulator.κ_DA - DA)
    return ϵ + modulator.κ_DA
end

function save_DA(sol, modulator, feedback, filepath, numtrial)
    DA = get_DA(sol, modulator, feedback)
    df = DataFrame(trial=numtrial, DA=DA)
    fname = "DA_values.csv"
    fullpath = joinpath(filepath, fname)
    
    # Append to existing file or create new one
    if isfile(fullpath)
        CSV.write(fullpath, df, append=true)
    else
        CSV.write(fullpath, df, writeheader=true)
    end
end

struct ProgressMeterMonitor <: AbstractExperimentMonitor
    meter::Progress
end
ProgressMeterMonitor(Ntrials::Int; kwargs...) = ProgressMeterMonitor(Progress(Ntrials; showspeed=true, kwargs...))

function NeurobloxBase.update_monitor!(monitor::ProgressMeterMonitor, sol, feedback, iscorrect, trace)
    next!(monitor.meter, showvalues=showvalues(;trace, N_trials=monitor.meter.n))
end

maybe_show_plot(args...; kwargs...) = []
showvalues(;trace, N_trials) = () -> begin
    
    pct_correct = round(sum(trace.correct)*100/length(trace.correct); digits=3)
    len=min(100, length(trace.correct))
    pct_correct_recent = let
        pct = sum(1:len) do i
            trace.correct[end-i+1]
        end / len
        (round(pct*100; digits=3))
    end
    N = 8
    last_N = map(min(length(trace.trial)-1, N-1):-1:0) do i
        trial = trace.trial[end-i]
        iscorrect = trace.correct[end-i]
        action = trace.action[end-i]
        time = trace.time[end-i]
        
        Response = iscorrect ? "\e[0;32mCorrect\e[0m," : "\e[0;31mFalse\e[0m,  "
        trial_str = rpad("$trial,", textwidth(string(N_trials))+1)

        DA_str = if haskey(trace, :DA)
            "DA = $(round(trace.DA[end-i], digits=2)), "
        else
            ""
        end
        
        ("Trial", "$trial_str Category choice = $(action), Response = $Response $(DA_str)Time = $(round(time, digits=1)) seconds")
    end
    [
        last_N
        ("Accuracy", "$(pct_correct)% total, $(pct_correct_recent)% last $len trials")
        maybe_show_plot(trace)
    ]
end

function runningperf(trace; len=min(150, length(trace.correct)))
    sum(1:len) do i
        trace.correct[end-i+1]
    end / len
end
