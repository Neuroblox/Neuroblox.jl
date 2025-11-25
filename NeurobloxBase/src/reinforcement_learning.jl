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
    maybe_set_state_pre!(lr::AbstractLearningRule, state)

If there is no previous value for the state, set it to `state`.
"""
function maybe_set_state_pre!(lr::AbstractLearningRule, state)
    if isnothing(lr.state_pre)
        lr.state_pre = state
    end
    lr
end

"""
    maybe_set_state_post!(lr::AbstractLearningRule, state)

If there is no post-learning value for the state, set it to `state`.
"""
function maybe_set_state_post!(lr::AbstractLearningRule, state)
    if isnothing(lr.state_post)
        lr.state_post = state
    end
    lr
end

maybe_set_state_pre!(lr::NoLearningRule, state) = lr
maybe_set_state_post!(lr::NoLearningRule, state) = lr

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
