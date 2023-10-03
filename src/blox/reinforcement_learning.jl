abstract type AbstractEnvironment end
abstract type AbstractLearningRule end

mutable struct HebbianPlasticity <:AbstractLearningRule
    const K
    const W_lim
    state_pre
    state_post

    function HebbianPlasticity(; K, W_lim, state_pre=nothing, state_post=nothing)
        new(K, W_lim, state_pre, state_post)
    end
end

function (hp::HebbianPlasticity)(sol::SciMLBase.AbstractSciMLSolution, t, w, feedback)
    val_pre, val_post = sol(t; idxs=[hp.state_pre, hp.state_post])
    Δw = hp.K * val_pre * val_post * (hp.W̄ - w) * feedback

    return Δw
end

mutable struct HebbianModulationPlasticity <: AbstractLearningRule
    const K
    const decay
    state_pre
    state_post
    modulator

    function HebbianModulationPlasticity(; 
        K, decay, 
        state_pre=nothing, state_post=nothing, modulator=nothing
    )
        new(K, decay, state_pre, state_post, modulator)
    end
end

function (hp::HebbianModulationPlasticity)(val_pre, val_modulator, w, feedback)
    DA = hp.modulator(val_modulator, feedback)
    DA_baseline = hp.modulator.κ_DA * hp.modulator.N_time_blocks

    Δw = feedback * hp.K * val_pre * DA * (DA - DA_baseline) * logistic(DA) - hp.decay * w

    return Δw
end

mutable struct ClassificationEnvironment <: AbstractEnvironment
    const stimulus
    const category
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

        new(stim, category, 1)
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

    function Agent(g::MetaDiGraph; name)
        sys = system_from_graph(g; name)
        policy = action_selection_from_graph(g)

        new(sys, policy)
    end
end
