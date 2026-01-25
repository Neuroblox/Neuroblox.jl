"""
    run_warmup(agent::AbstractAgent, env::AbstractEnvironment, t_warmup; kwargs...)

Run the initial solve of the RL experiment for `t_warmup`.
"""
function run_warmup end

"""
    run_trial!(agent::AbstractAgent, env::AbstractEnvironment, weights, u0; kwargs...)

Run a single trial of a RL experiment. Update the connection weights according
to the learning rules.
"""
function run_trial! end

"""
    run_experiment!(agent::AbstractAgent, env::AbstractEnvironment; verbose=false, t_warmup=0, kwargs...)

Perform a full RL experiment with `agent` in the environment `env`. Will run until the maximum number of trials in `env` is reached.
"""
function run_experiment! end

"""
    get_trial_stimulus(env::AbstractEnvironment)

Get a dictionary mapping stimulus parameters to their values for the current trial.
"""
function get_trial_stimulus end

"""
    connect_action_selection!(as::AbstractActionSelection, blox_src, blox_dst)

Given a policy, set the list of actions that the agent can choose between next.
"""
function connect_action_selection! end

"""
    weight_gradient(lr::AbstractLearningRule, sol, w, feedback)

Calculate the way that the weight `w` should change based on the solution of the 
reinforcement learning experiment.
"""
function weight_gradient end

"""
    maybe_set_state_pre(lr::AbstractLearningRule, state)

If there is no previous value for the state, make a copy of the learning rule where `lr.state_pre == state` 
"""
function maybe_set_state_pre(lr::AbstractLearningRule, state)
    if isnothing(lr.state_pre)
        @set lr.state_pre = state
    else
        lr
    end
end

"""
    maybe_set_state_post(lr::AbstractLearningRule, state)

If there is no post-learning value for the state, make a copy of the learning rule where `lr.state_post == state` 
"""
function maybe_set_state_post(lr::AbstractLearningRule, state)
    if isnothing(lr.state_post)
        @set lr.state_post = state
    else
        lr
    end
end
maybe_set_state_pre(lr::NoLearningRule, state) = lr
maybe_set_state_post(lr::NoLearningRule, state) = lr

increment_trial!(env::AbstractEnvironment) = env.current_trial = mod(env.current_trial, env.N_trials) + 1

reset!(env::AbstractEnvironment) = env.current_trial = 1


"""
    update_monitor!(monitor, trial_solution, feedback, iscorrect, trace)

Overload this function to support custom `monitor` objects passed to `run_experiment!`. After every trial in the experiment,
the function `update_monitor` will be called where
* trial_solution is the `ODESolution` to the most recent `trial`
* `feedback` is the feedback prediction made by the model (i.e. category choice)
* `iscorrect` is a `Bool` saying if the feedback was correct
* `trace` is a `@NamedTuple{trial::Vector{Int}, iscorrect::Vector{Bool}, action::Vector{Int}, time::Vector{Float64}}` with the history of all the previous trials containing info on whether they were correct, which feedback action was chosen, and how long the trial took.
"""
function update_monitor! end

function update_monitor!(monitor::Nothing, trial_solution, feedback, iscorrect, trace)
    nothing
end

function action_selection_from_graph(g::GraphSystem)
    sels = [blox for blox in nodes(g) if blox isa AbstractActionSelection]
    if isempty(sels)
        @warn "No action selection provided"
    elseif length(sels) == 1
        sel = only(sels)
        srcs = []
        for (;src, dst) ∈ connections(g)
            if dst == sel
                push!(srcs, src)
            end
        end
        if length(srcs) != 2
            error("Two blocks need to connect to the action selection $(sel.name) block")
        end
        connect_action_selection!(sel, srcs[1], srcs[2])
        sel
    else
        error("Multiple action selection blocks are detected. Only one must be used in an experiment.")
    end
end

function t_block_event(key)
    function _apply_t_block_event!(integrator)
        (; params_partitioned, connection_matrices, partition_plan) = integrator.p
        states_partitioned = partitioned(integrator.u, partition_plan)
        t = integrator.t
        # Some t_block events need to happen before others, so we split them into two categories: 'early' and 'late'.
        # the old implementation did this as separate events, but here we can just force the early ones to happen before
        # the late ones without having to have extra events.

        for i ∈ eachindex(states_partitioned)
            states_partitioned_i = states_partitioned[i]
            params_partitioned_i = params_partitioned[i]
            tag = get_tag(eltype(states_partitioned_i))
            if has_t_block_event(tag)
                if is_t_block_event_time(tag, key, t)
                    for j ∈ eachindex(states_partitioned_i)
                        sys_dst = Subsystem(states_partitioned_i[j], params_partitioned_i[j])
                        if t_block_event_requires_inputs(tag)
                            input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)
                        else
                            input = initialize_input(sys_dst)
                        end
                        apply_t_block_event!(@view(states_partitioned_i[j]), @view(params_partitioned_i[j]), sys_dst, input, t)
                    end
                end
            end
        end
    end
end

has_t_block_event(::Type{Union{}}) = error("Something went wrong. This error should only exist for method disambiguation")
t_block_event_requires_inputs(::Type{Union{}}) = error("Something went wrong. This error should only exist for method disambiguation")
has_t_block_event(::Type{T}) where {T} = false

function apply_learning_rules!(sol, prob, learning_rules, feedback)
    (;connection_matrices, params_partitioned, names_partitioned) = prob.p
    for ((;nc, k, i, l, j), rule) ∈ learning_rules
        conn = connection_matrices[nc][k,i][l,j]
        Δw = weight_gradient(rule, sol, conn.weight, feedback)
        name_dst = names_partitioned[i][j]
        name_src = names_partitioned[k][l]
        if !isfinite(Δw)
            @warn "non-finite gradient" name_dst name_src Δw
        end
        connection_matrices[nc][k,i][l,j]= @reset conn.weight += (Δw)
    end
end
