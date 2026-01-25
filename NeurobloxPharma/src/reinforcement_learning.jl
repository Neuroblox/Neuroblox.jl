"""
    HebbianPlasticity(; K, W_lim,
                      state_pre = nothing,
                      state_post = nothing,
                      t_pre = nothing,
                      t_post = nothing)

Hebbian plasticity rule. The connection weight is updated according to :

```math
    w_{j+1} = w_j + \\text{feedback} × K x_\\text{pre} x_\\text{post} (W_\\text{lim} - w)
```
where `feedback` is a binary indicator of the correctness of the model's action, 
and `x` indicates the activity of the pre- and post-synaptic neuron states `state_pre` and `state_post` at timepoints `t_pre` and `t_post` respectively.

Arguments:
    - K : the learning rate of the connection
    - W_lim : the maximum weight for the connection
    - state_pre : state of the presynaptic neuron that is used in the plasticity rule (by default this is state `V` in neurons).
    - state_post : state_pre : state of the postsynaptic neuron that is used in the plasticity rule (by default this is state `V` in neurons). 
    - t_pre : timepoint at which `state_pre` is evaluated to be used in the plasticity rule.
    - t_post : t_pre : timepoint at which `state_post` is evaluated to be used in the plasticity rule.

See also [`HebbianModulationPlasticity`](@ref).
"""
mutable struct HebbianPlasticity <:AbstractLearningRule
    const K::Float64
    const W_lim::Float64
    state_pre::Union{Nothing, Symbol}
    state_post::Union{Nothing, Symbol}
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

function NeurobloxBase.weight_gradient(hp::HebbianPlasticity, sol, w, feedback)
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

Hebbian plasticity rule, modulated by the dopamine reward prediction error. The weight update is largest when the reward prediction error is far from the modulation threshold `θₘ`.

```math
    ϵ = \\text{feedback} - (\\text{DA}_b - \\text{DA})
    w_{j+1} = w_j + \\max(\\times K x_\\text{pre} x_\\text{post} ϵ(ϵ + θₘ) dσ(α(ϵ + θₘ)) - \\text{decay} × w, -w)
```
where `feedback` is a binary indicator of the correctness of the model's action, 
`DA_b` is the baseline dopamine level, `DA` is the modulator's dopamine release, 
`dσ` is the derivative of the logistic function,
and `x` indicates the activity of the pre- and post-synaptic neuron states `state_pre` and `state_post` at timepoints `t_pre` and `t_post` respectively. 
The decay prevents the weights from diverging.

Arguments:
    - K: the learning rate of the connection
    - decay: Decay of the weight update
    - α: the selectivity of the derivative of the logistic function
    - θₘ: the modulation threshold for the reward prediction error
    - state_pre : state of the presynaptic neuron that is used in the plasticity rule (by default this is state `V` in neurons).
    - state_post : state_pre : state of the postsynaptic neuron that is used in the plasticity rule (by default this is state `V` in neurons). 
    - t_pre : timepoint at which `state_pre` is evaluated to be used in the plasticity rule.
    - t_post : t_pre : timepoint at which `state_post` is evaluated to be used in the plasticity rule.

See also [`HebbianPlasticity`](@ref).
"""
mutable struct HebbianModulationPlasticity <: AbstractLearningRule
    const K::Float64
    const decay::Float64
    const α::Float64
    const θₘ::Float64
    state_pre::Union{Nothing, Symbol}
    state_post::Union{Nothing, Symbol}
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

function NeurobloxBase.weight_gradient(hmp::HebbianModulationPlasticity, sol, w, feedback)
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
    const source::S
    const category::Vector{Int}
    const N_trials::Int
    const t_trial::Float64
    current_trial::Int
    
    function ClassificationEnvironment(data::DataFrame; t_stimulus, t_pause)
        stim = ImageStimulus(
                        data; 
                        name=:stim, 
                        namespace=namespaced_name(namespace, name),
                        t_stimulus,
                        t_pause
        )
        
        N_trials = stim.N_stimuli

        ClassificationEnvironment(stim, N_trials)
    end

    function ClassificationEnvironment(data::DataFrame, N_trials; t_stimulus, t_pause)
        stim = ImageStimulus(
                        data; 
                        name=:stim, 
                        namespace=namespaced_name(namespace, name),
                        t_stimulus,
                        t_pause
        )

        ClassificationEnvironment(stim, N_trials)
    end
    
    function ClassificationEnvironment(stim::ImageStimulus)
        N_trials = stim.param_vals.N_stimuli

        ClassificationEnvironment(stim, N_trials)
    end

    function ClassificationEnvironment(stim::ImageStimulus, N_trials)
        t_trial = stim.param_vals.t_stimulus + stim.param_vals.t_pause

        new{typeof(stim)}(stim, stim.param_vals.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]

function NeurobloxBase.get_trial_stimulus(env::ClassificationEnvironment)
    stim_params = env.source.stim_parameters
    stim_values = env.source.IMG[:, env.current_trial]

    return Dict(p => v for (p, v) in zip(stim_params, stim_values))
end

"""
    GreedyPolicy(; name, t_decision, namespace, competitor_states = Symbol[])

A policy that makes a choice by picking the state with the highest value among `competitor_states` which represent each available choice. `t_decision` is the time of the decision.
"""
mutable struct GreedyPolicy <: AbstractActionSelection
    const name::Symbol
    const namespace::Symbol
    competitor_states::Vector{Symbol}
    competitor_params::Vector{Symbol}
    const t_decision::Float64

    function GreedyPolicy(; name, t_decision, namespace=nothing, competitor_states=Symbol[], competitor_params=Symbol[])
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

struct Agent{S,P,A,LR,CM} <: AbstractAgent
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connection_matrices::CM
end

"""
    Agent(g::GraphSystem; t_block=missing, u0=[], p=[], kwargs...)

A reinforcement learning agent, used to interact with an AbstractEnvironment to simulate a learning task.

Arguments : 
- g : A GraphSystem containing the model that the agent is using to make choices and update its connections during reinforcement learning.

Keyword arguments : 
- u0 : Initial conditions for the model in g. If not provided then default values will be used.
- p : Parameter values for the model in g. If not provided then default values will be used.
- t_block : The time period of a PeriodicCallback which will reset the cumulative spike counter of neurons in the model. This is optional and can be useful when the plasticity rules require number of spikes within specific time windows to update connection weights.
- kwargs... : All other keyword arguments are passed to the ODEProblem that is constructed inside the agent and solved during run_experiment! and run_trial! . 
"""
function Agent(g::GraphSystem; t_block=missing, u0=[], p=[], kwargs...)
    if !ismissing(t_block)
        global_events=[PeriodicCallback(t_block_event(:t_block_early), t_block - √(eps(float(t_block)))),
                       PeriodicCallback(t_block_event(:t_block_late), t_block  +2*√(eps(float(t_block))))]
    else
        global_events=[]
    end
    prob = ODEProblem(g, u0, (0.,1.), p; global_events, kwargs...)
    policy = action_selection_from_graph(g)
    learning_rules = GraphDynamics.OrderedDict(((; nc, k,i,j,l) => kwargs.learning_rule for (; kwargs, nc,k,i,j,l) ∈ connections(g.flat_graph)
                                      if get(kwargs, :learning_rule, NoLearningRule()) != NoLearningRule()))
    conn = prob.p.connection_matrices
    Agent(g, prob, policy, learning_rules, conn)
end

function NeurobloxBase.run_experiment!(agent::Agent, env::ClassificationEnvironment, save_path::Union{Nothing, String}=nothing, blocks=nothing;
                                       t_warmup=0,
                                       monitor=nothing,
                                       interrupt_token = Threads.Atomic{Bool}(false),
                                       modulator=nothing,
                                       kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)
    if t_warmup > 0
        u0 = @noinline run_warmup(agent, env, t_warmup; kwargs...)
        @reset agent.problem = remake(agent.problem; tspan, u0=u0)
    else
        @reset agent.problem = remake(agent.problem; tspan)
    end
    
    if isnothing(modulator)
        trace = (; trial=Int[], correct=Bool[], action=Int[], time=Float64[])
    else
        trace = (; trial=Int[], correct=Bool[], action=Int[], time=Float64[], DA=Float64[])
    end
    try
        for trial ∈ 1:N_trials
            if interrupt_token[]
                @warn "Interrupted! Bailing out now"
                break
            end
            (;time, gctime) = @timed begin
                sol, iscorrect, action = @noinline run_trial!(agent, env; kwargs...)
            end
            push!(trace.trial, trial)
            push!(trace.correct, iscorrect)
            push!(trace.action, action)
            push!(trace.time, time)
            if !isnothing(modulator)
                push!(trace.DA, get_DA(sol, modulator, iscorrect))
            end
            update_monitor!(monitor, sol, action, iscorrect, trace)
            if !isnothing(save_path)
                save_voltages(sol, save_path, trial)
                if !isnothing(blocks)
                    save_voltages_block(sol, save_path, trial, blocks)
                end
                if !isnothing(modulator)
                    save_DA(sol, modulator, iscorrect, save_path, trial)
                end
            end
        end
    catch e;
        if e isa InterruptException
            @warn "Interrupted! Bailing out now"
        else
            rethrow(e)
        end
    end
    trace
end

function NeurobloxBase.run_warmup(agent::Agent, env::ClassificationEnvironment, t_warmup; alg, kwargs...)
    prob = remake(agent.problem; tspan=(0, t_warmup))
    sol = solve(prob, alg; save_everystep=false, kwargs...)
    u0 = sol[:,end] # last value of state vector
    return u0
end

function NeurobloxBase.run_trial!(agent::Agent, env; alg, kwargs...)
    prob = agent.problem
    action_selection = agent.action_selection
    learning_rules = agent.learning_rules

    update_trial_stimulus!(prob, env)
    
    sol = solve(prob, alg; kwargs...)
    if isnothing(action_selection)
        feedback = 1
        action = 0
    else
        action = action_selection(sol)
        feedback = env(action)
    end
    apply_learning_rules!(sol, prob, learning_rules, feedback)
    increment_trial!(env)
    
    return sol, feedback, action
end

function update_trial_stimulus!(prob, env::ClassificationEnvironment)
    (;params_partitioned) = prob.p
    for i ∈ eachindex(params_partitioned)
        if eltype(params_partitioned[i]) <: SubsystemParams{ImageStimulus}
            stim = only(params_partitioned[i])
            stim.current_image .= stim.IMG[:, env.current_trial]
        end
    end
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
