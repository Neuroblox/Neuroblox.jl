abstract type AbstractEnvironment end
abstract type AbstractLearningRule end

mutable struct HebbianPlasticity <:AbstractLearningRule
    const K
    const W_lim
    state_pre
    state_post
    t_pre
    t_post

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

mutable struct HebbianModulationPlasticity <: AbstractLearningRule
    const K
    const decay
    const α
    const θₘ
    state_pre
    state_post
    t_pre
    t_post
    t_mod
    modulator

    function HebbianModulationPlasticity(; 
        K, decay, α, θₘ, modulator=nothing,
        state_pre=nothing, state_post=nothing, 
        t_pre=nothing, t_post=nothing, t_mod=nothing,   
    )
        new(K, decay, α, θₘ, state_pre, state_post, t_pre, t_post, t_mod, modulator)
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

        category = data[!, :category]
        N_trials = stim.N_stimuli
        t_trial = t_stimulus + t_pause

        new{typeof(stim)}(Symbol(name), Symbol(namespace), stim, category, N_trials, t_trial, 1)
    end

    function ClassificationEnvironment(stim::ImageStimulus; name, namespace=nothing)
        t_trial = stim.t_stimulus + stim.t_pause
        N_trials = stim.N_stimuli

        new{typeof(stim)}(Symbol(name), Symbol(namespace), stim, stim.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]

increment_trial!(env::AbstractEnvironment) = env.current_trial += 1

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

"""
function (p::GreedyPolicy)(sys::ODESystem, prob::ODEProblem)
    ps = parameters(sys)
    params = prob.p
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))
    comp_params = p.competitor_params
    idxs_cp = Int64[]
    for i in eachindex(comp_params)
        idxs = findall(x -> x==comp_params[i], ps)
        push!(idxs_cp,idxs)
    end
    comp_vals = params[map_idxs[idxs_cp]]
    @info comp_vals
    return argmax(comp_vals)
end
"""

mutable struct Agent{S,P,A,LR,PA}
    odesystem::S
    problem::P
    action_selection::A
    learning_rules::LR
    init_params::PA
    # simsys::SS

    function Agent(g::MetaDiGraph; name, kwargs...)
        bc = connector_from_graph(g)

        t_block = haskey(kwargs, :t_block) ? kwargs[:t_block] : missing
        # TODO: add another version that uses system_from_graph(g,bc,params;)
        sys = system_from_graph(g, bc; name, t_block, allow_parameter=false)

        u0 = haskey(kwargs, :u0) ? kwargs[:u0] : []
        p = haskey(kwargs, :p) ? kwargs[:p] : []
        
        prob = ODEProblem(sys, u0, (0.,1.), p)
        init_params = copy(prob.p)
        
        policy = action_selection_from_graph(g)
        learning_rules = bc.learning_rules

        new{typeof(sys), typeof(prob), typeof(policy), typeof(learning_rules), typeof(init_params)#=, typeof(ss)=#}(sys, prob, policy, learning_rules, init_params, #=ss=#)
    end
end

reset!(ag::Agent) = ag.problem = remake(ag.problem; p = ag.init_params)

function run_experiment!(agent::Agent, env::ClassificationEnvironment, t_warmup=200.0; kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)

    sys = get_sys(agent)
    prob = agent.problem

    if t_warmup > 0
        prob = remake(prob; tspan=(0,t_warmup))
        if haskey(kwargs, :alg)
            sol = solve(prob, kwargs[:alg]; kwargs...)
        else
            sol = solve(prob; alg_hints = [:stiff], kwargs...)
        end
        u0 = sol[1:end,end] # last value of state vector
        prob = remake(prob; tspan=tspan, u0=u0)
    else
        prob = remake(prob; tspan)
        u0 = []
    end

    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    
    defs = ModelingToolkit.get_defaults(sys)
    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

    for _ in Base.OneTo(N_trials)

        stim_params = get_trial_stimulus(env)

        to_update = merge(weights, stim_params)
        new_params = ModelingToolkit.MTKParameters(sys, merge(defs, weights, stim_params))

        prob = remake(prob; p = new_params, u0=u0)
        if haskey(kwargs, :alg)
            sol = solve(prob, kwargs[:alg]; kwargs...)
        else
            sol = solve(prob; alg_hints = [:stiff], kwargs...)
        end

        # u0 = sol[1:end,end] # next run should continue where the last one ended   
        # In the paper we assume sufficient time interval before net stimulus so that
        # system reaches back to steady state, so we don't continue from previous trial's endpoint

        if isnothing(action_selection)
            feedback = 1
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
    end

    agent.problem = prob
end

function run_trial!(agent::Agent, env::ClassificationEnvironment, weights::Dict{Num, Float64}, u0::Vector{Float64}; kwargs...)
    N_trials = env.N_trials

    if env.current_trial <= N_trials
        t_trial = env.t_trial
        tspan = (0, t_trial)

        prob = agent.problem

        action_selection = agent.action_selection
        learning_rules = agent.learning_rules
        
        @show env.current_trial
        stim_params = get_trial_stimulus(env)
        @show stim_params
        @show weights
        prob = remake(prob; tspan=tspan, p = merge(weights, stim_params), u0=u0)

        if haskey(kwargs, :alg)
            sol = solve(prob, kwargs[:alg]; kwargs...)
        else
            sol = solve(prob; alg_hints = [:stiff], kwargs...)
        end
        
        if isnothing(action_selection)
            feedback = 1
        else
            action = action_selection(sol)
            feedback = env(action)
        end

        for (w, rule) in learning_rules
            w_val = weights[w]
            Δw = weight_gradient(rule, sol, w_val, feedback)
            @show Δw
            weights[w] += Δw
        end
        prob = remake(prob; p = merge(weights)) #updates the weights in prob
        increment_trial!(env)
        agent.problem = prob
       # u0 = sol[1:end,end]
    end
end

function run_experiment!(agent::Agent, env::ClassificationEnvironment, save_path::String, t_warmup=200.0; kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)
    sys = get_sys(agent)
    prob = agent.problem

    if t_warmup > 0
        prob = remake(prob; tspan=(0,t_warmup))
        if haskey(kwargs, :alg)
            sol = solve(prob, kwargs[:alg]; kwargs...)
        else
            sol = solve(prob; alg_hints = [:stiff], kwargs...)
        end
        u0 = sol[1:end,end] # last value of state vector
        prob = remake(prob; tspan=tspan, u0=u0)
    else
        prob = remake(prob; tspan)
        u0 = []
    end

    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    
    defs = ModelingToolkit.get_defaults(sys)
    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

    for trial_num in Base.OneTo(N_trials)

        stim_params = get_trial_stimulus(env)

        to_update = merge(weights, stim_params)
        new_params = ModelingToolkit.MTKParameters(sys, merge(defs, weights, stim_params))

        prob = remake(prob; p = new_params, u0=u0)
        if haskey(kwargs, :alg)
            sol = solve(prob, kwargs[:alg]; kwargs...)
        else
            sol = solve(prob; alg_hints = [:stiff], kwargs...)
        end

        # u0 = sol[1:end,end] # next run should continue where the last one ended   
        # In the paper we assume sufficient time interval before net stimulus so that
        # system reaches back to steady state, so we don't continue from previous trial's endpoint

        if isnothing(action_selection)
            feedback = 1
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

        if !isnothing(save_path)
            save_voltages(sol, save_path, trial_num)
        end

    end

    agent.problem = prob
end

function save_voltages(sol, filepath, numtrial)
    df = DataFrame(sol)
    fname = "sim"*lpad(numtrial, 4, "0")*".csv"
    fullpath = joinpath(filepath, fname)
    CSV.write(fullpath, df)
end
