abstract type AbstractEnvironment end
abstract type AbstractLearningRule end

struct NoLearningRule <: AbstractLearningRule end

mutable struct HebbianPlasticity <:AbstractLearningRule
    const K::Float64
    const W_lim::Float64
    state_pre::Union{Nothing, Num}
    state_post::Union{Nothing, Num}
    t_pre::Float64
    t_post::Float64

    function HebbianPlasticity(; 
        K, W_lim, 
        state_pre=nothing, state_post=nothing,
        t_pre=nothing, t_post=nothing
    )
        new(K, W_lim, state_pre, state_post, t_pre, t_post)
    end
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

mutable struct HebbianModulationPlasticity{M} <: AbstractLearningRule
    const K::Float64
    const decay::Float64
    const α::Float64
    const θₘ::Float64
    state_pre::Union{Nothing, Num}
    state_post::Union{Nothing, Num}
    t_pre::Float64
    t_post::Float64
    t_mod::Float64
    modulator::M

    function HebbianModulationPlasticity(; 
        K, decay, α, θₘ, modulator=nothing,
        state_pre=nothing, state_post=nothing, 
        t_pre=nothing, t_post=nothing, t_mod=nothing,   
    )
        new{typeof(modulator)}(K, decay, α, θₘ, state_pre, state_post, t_pre, t_post, t_mod, modulator)
    end
end

dlogistic(x) = logistic(x) * (1 - logistic(x)) 

function (hmp::HebbianModulationPlasticity)(val_pre, val_post, val_modulator, w, feedback)
    DA = hmp.modulator(val_modulator)
    DA_baseline = hmp.modulator.κ_DA * hmp.modulator.N_time_blocks
    ϵ = feedback - (hmp.modulator.κ_DA - DA)
    
   # Δw = hmp.K * val_post * val_pre * DA * (DA - DA_baseline) * dlogistic(DA) - hmp.decay * w
    Δw = maximum([hmp.K * val_post * val_pre * ϵ * (ϵ + hmp.θₘ) * dlogistic(hmp.α * (ϵ + hmp.θₘ)) - hmp.decay * w, -w])

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

function maybe_set_state_pre!(lr::AbstractLearningRule, state)
    if isnothing(lr.state_pre)
        lr.state_pre = state
    end
end

function maybe_set_state_post!(lr::AbstractLearningRule, state)
    if isnothing(lr.state_post)
        lr.state_post = state
    end
end

maybe_set_state_pre!(lr::NoLearningRule, state) = lr
maybe_set_state_post!(lr::NoLearningRule, state) = lr

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

increment_trial!(env::AbstractEnvironment) = env.current_trial = mod(env.current_trial, env.N_trials) + 1

reset!(env::AbstractEnvironment) = env.current_trial = 1

function get_trial_stimulus(env::ClassificationEnvironment)
    stim_params = env.source.stim_parameters
    stim_values = env.source.IMG[:, env.current_trial]

    return Dict(p => v for (p, v) in zip(stim_params, stim_values))
end

abstract type AbstractActionSelection <: AbstractBlox end

mutable struct GreedyPolicy <: AbstractActionSelection
    const name::Symbol
    const namespace::Symbol
    competitor_states::Vector{Num}
    competitor_params::Vector{Num}
    const t_decision::Float64

    function GreedyPolicy(; name, t_decision, namespace=nothing, competitor_states=nothing, competitor_params=nothing)
        sts = isnothing(competitor_states) ? Num[] : competitor_states
        ps = isnothing(competitor_states) ? Num[] : competitor_params
        new(name, namespace, sts, ps, t_decision)
    end
end

function (p::GreedyPolicy)(sol::SciMLBase.AbstractSciMLSolution)
    comp_vals = sol(p.t_decision; idxs=p.competitor_states)
    return argmax(comp_vals)
end

function connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, get_matrisome(str1), get_matrisome(str2))
end

function connect_action_selection!(as::AbstractActionSelection, matr1::Matrisome, matr2::Matrisome)
    sys1 = get_namespaced_sys(matr1)
    sys2 = get_namespaced_sys(matr2)

    as.competitor_states = [sys1.ρ_, sys2.ρ_] #HACK : accessing values of rho at a specific time after the simulation
    #as.competitor_params = [sys1.H, sys2.H]
end

get_eval_times(gp::GreedyPolicy) = [gp.t_decision]

get_eval_states(gp::GreedyPolicy) = gp.competitor_states

mutable struct Agent{S,P,A,LR,C}
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connector::C

    function Agent(g::MetaDiGraph; name, kwargs...)
        conns = connectors_from_graph(g)
        
        t_block = haskey(kwargs, :t_block) ? kwargs[:t_block] : missing
        # TODO: add another version that uses system_from_graph(g,bc,params;)
        sys = system_from_graph(g, conns; name, t_block, allow_parameter=false)

        u0 = haskey(kwargs, :u0) ? kwargs[:u0] : []
        p = haskey(kwargs, :p) ? kwargs[:p] : []
        
        prob = ODEProblem(sys, u0, (0.,1.), p)
        
        policy = action_selection_from_graph(g)
        lr =  narrowtype(learning_rules(conns))  

        new{typeof(sys), typeof(prob), typeof(policy), typeof(lr), typeof(conns)}(sys, prob, policy, lr, conns)
    end
end

function run_experiment!(agent::Agent, env::ClassificationEnvironment; verbose=false, t_warmup=0, kwargs...)
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

    t_stops = mapreduce(get_eval_times, union, values(learning_rules); init=Float64[])

    action_selection = agent.action_selection 
    if !isnothing(action_selection)
        t_stops = union(t_stops, get_eval_times(action_selection))
    end

    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

    trace = NamedTuple{(:trial, :correct, :action)}((Int[], Bool[], Int[]))

    for trial in Base.OneTo(N_trials)
        _, iscorrect, action= run_trial!(agent, env, weights, nothing; kwargs...)

        push!(trace.trial, trial)
        push!(trace.correct, iscorrect)
        push!(trace.action, action)

        if verbose
            println("Trial = $(trial), Category choice = $(action), Response = $(iscorrect==1 ? "Correct" : "False")")
        end
    end
    return trace
end

function run_experiment!(agent::Agent, env::ClassificationEnvironment, save_path::String; verbose=false, t_warmup=0, kwargs...)
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

    trace = NamedTuple{(:trial, :correct, :action)}((Int[], Bool[], Int[]))

    for trial in Base.OneTo(N_trials)
        sol, iscorrect, action= run_trial!(agent, env, weights, nothing; kwargs...)

        save_voltages(sol, save_path, trial)

        push!(trace.trial, trial)
        push!(trace.correct, iscorrect)
        push!(trace.action, action)

        if verbose
            println("Trial = $(trial), Category choice = $(action), Response = $(iscorrect==1 ? "Correct" : "False")")
        end
    end

    return trace
end

function run_warmup(agent::Agent, env::ClassificationEnvironment, t_warmup; kwargs...)

    prob = remake(agent.problem; tspan=(0, t_warmup))
    if haskey(kwargs, :alg)
        sol = solve(prob, kwargs[:alg]; save_everystep=false, kwargs...)
    else
        sol = solve(prob; alg_hints = [:stiff], save_everystep=false, kwargs...)
    end
    u0 = sol[:,end] # last value of state vector

    return u0
end

function run_trial!(agent::Agent, env::ClassificationEnvironment, weights, u0; kwargs...)

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
