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
