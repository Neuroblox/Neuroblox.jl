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
    Δw = hp.K * val_pre * val_post * (hp.W̄ - w) * feedback

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
        new(K, decay, modulator, state_pre, state_post, t_pre, t_post, t_mod)
    end
end

function (hmp::HebbianModulationPlasticity)(val_pre, val_post, val_modulator, w, feedback)
    DA = hmp.modulator(val_modulator, feedback)
    DA_baseline = hmp.modulator.κ_DA * hmp.modulator.N_time_blocks

    Δw = hmp.K * val_post * val_pre * DA * (DA - DA_baseline) * derivative_logistic(DA) - hmp.decay * w

    return Δw
end

function weight_gradient(hmp::HebbianModulationPlasticity, sol, w, feedback)
    state_mod = get_modulator_state(hmp.modulator)
    val_pre = only(sol(hmp.t_pre; idxs = [hmp.state_pre]))
    val_post = only(sol(hmp.t_post; idxs = [hmp.state_post]))
    val_mod = only(sol(hmp.t_mod; idxs = [state_mod]))

    return hmp(val_pre, val_post, val_mod, w, feedback)
end

mutable struct ClassificationEnvironment <: AbstractEnvironment
    const name
    const namespace
    const stimulus
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
        N_trials = DataFrames.nrow(data)
        t_trial = t_stimulus + t_pause

        new(name, namespace, stim, category, N_trials, t_trial, 1)
    end

    function ClassificationEnvironment(stim::ImageStimulus; name, namespace=nothing, t_stimulus, t_pause)
        t_trial = t_stimulus + t_pause
        N_trials = size(stim.image)[2]

        new(name, namespace, stim, stim.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]

increment_trial!(env::AbstractEnvironment) = env.current_trial += 1

abstract type AbstractActionSelection end

mutable struct GreedyPolicy <: AbstractActionSelection
    competitor_states 
    const t_decision

    function GreedyPolicy(; competitor_states=nothing, t_decision)
        sts = isnothing(competitor_states) ? Num[] : competitor_states
        new(sts, t_decision)
    end
end

function (p::GreedyPolicy)(sol::SciMLBase.AbstractSciMLSolution)
    comp_vals = sol(p.t_decision; idxs=[p.competitors])
    return argmax(comp_vals)
end
 
struct Agent 
    odesystem
    action_selection
    learning_rules

    function Agent(g::MetaDiGraph; name)
        bc = connector_from_graph(g)

        sys = system_from_graph(g, bc; name)
        ssys = structural_simplify(sys)

        policy = action_selection_from_graph(g)
        learning_rules = bc.learning_rules

        new(ssys, policy, learning_rules)
    end
end

    end
end
