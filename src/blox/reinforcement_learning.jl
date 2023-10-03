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

