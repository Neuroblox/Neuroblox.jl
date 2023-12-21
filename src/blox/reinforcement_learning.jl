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
    state_pre
    state_post
    t_pre
    t_post
    t_mod
    modulator

    function HebbianModulationPlasticity(; 
        K, decay, modulator=nothing,
        state_pre=nothing, state_post=nothing, 
        t_pre=nothing, t_post=nothing, t_mod=nothing,   
    )
        new(K, decay, state_pre, state_post, t_pre, t_post, t_mod, modulator)
    end
end

dlogistic(x) = logistic(x) * (1 - logistic(x)) 

function (hmp::HebbianModulationPlasticity)(val_pre, val_post, val_modulator, w, feedback)
    DA = hmp.modulator(val_modulator, feedback)
    DA_baseline = hmp.modulator.κ_DA * hmp.modulator.N_time_blocks

    Δw = hmp.K * val_post * val_pre * DA * (DA - DA_baseline) * dlogistic(DA) - hmp.decay * w

    return Δw
end

function weight_gradient(hmp::HebbianModulationPlasticity, sol, w, feedback)
    state_mod = get_modulator_state(hmp.modulator)
    val_pre = only(sol(hmp.t_pre; idxs = [hmp.state_pre]))
    val_post = only(sol(hmp.t_post; idxs = [hmp.state_post]))
    val_mod = only(sol(hmp.t_mod; idxs = [state_mod]))

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

mutable struct ClassificationEnvironment <: AbstractEnvironment
    const name
    const namespace
    const source
    const category
    const N_trials
    const t_trial
    current_trial

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

        new(name, namespace, stim, category, N_trials, t_trial, 1)
    end

    function ClassificationEnvironment(stim::ImageStimulus; name, namespace=nothing)
        t_trial = stim.t_stimulus + stim.t_pause
        N_trials = stim.N_stimuli

        new(name, namespace, stim, stim.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]

increment_trial!(env::AbstractEnvironment) = env.current_trial += 1

function get_trial_stimulus(env::ClassificationEnvironment)
    stim_params = env.source.stim_parameters
    stim_values = env.source.IMG[:, env.current_trial]

    return Dict(p => v for (p, v) in zip(stim_params, stim_values))
end

abstract type AbstractActionSelection <: AbstractBlox end

mutable struct GreedyPolicy <: AbstractActionSelection
    const name
    const namespace
    competitor_states 
    const t_decision

    function GreedyPolicy(; name, t_decision, namespace=nothing, competitor_states=nothing)
        sts = isnothing(competitor_states) ? Num[] : competitor_states
        new(name, namespace, sts, t_decision)
    end
end

function (p::GreedyPolicy)(sol::SciMLBase.AbstractSciMLSolution)
    comp_vals = sol(p.t_decision; idxs=p.competitor_states)
    return argmax(comp_vals)
end
 
mutable struct Agent 
    odesystem
    problem
    action_selection
    learning_rules

    function Agent(g::MetaDiGraph; name, kwargs...)
        bc = connector_from_graph(g)

        t_affect = haskey(kwargs, :t_block) ? kwargs[:t_block] : missing
        sys = system_from_graph(g, bc; name, t_affect)
        ss = structural_simplify(sys; allow_parameter=false)

        u0 = haskey(kwargs, :u0) ? kwargs[:u0] : []
        p = haskey(kwargs, :p) ? kwargs[:p] : []
        
        prob = ODEProblem(ss, u0, (0,1), p)

        policy = action_selection_from_graph(g)
        learning_rules = bc.learning_rules

        new(ss, prob, policy, learning_rules)
    end
end

function run_experiment!(agent::Agent, env::ClassificationEnvironment; kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)

    sys = get_sys(agent)
    prob = agent.problem
    prob = remake(prob; tspan)

    action_selection = agent.action_selection
    learning_rules = agent.learning_rules
    
    defs = ModelingToolkit.get_defaults(sys)
    weights = Dict{Num, Float64}()
    for w in keys(learning_rules)
        weights[w] = defs[w]
    end

    for _ in Base.OneTo(N_trials)
        stim_params = get_trial_stimulus(env)
        prob = remake(prob; p = merge(weights, stim_params))

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
            weights[w] += Δw
        end

        increment_trial!(env)
    end

    agent.problem = prob
end
