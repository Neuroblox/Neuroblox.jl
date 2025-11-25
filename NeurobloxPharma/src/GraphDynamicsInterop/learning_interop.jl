struct GDYAgent{S,P,A,LR,CM} <: AbstractAgent
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connection_matrices::CM
end

function Agent(g::GraphSystem; name, t_block=missing, u0=[], p=[], graphdynamics=true, kwargs...)
    if !graphdynamics
        return Agent(to_metadigraph(g); name, t_block, u0, p, kwargs...)
    end
    
    if !ismissing(t_block)
        global_events=[PeriodicCallback(t_block_event(:t_block_early), t_block - √(eps(float(t_block)))),
                       PeriodicCallback(t_block_event(:t_block_late), t_block  +2*√(eps(float(t_block))))]
    else
        global_events=[]
    end
    sys_par = PartitionedGraphSystem(g)
    prob = ODEProblem(sys_par, u0, (0.,1.), p; global_events, kwargs...)
    policy = action_selection_from_graph(g)
    learning_rules = make_connection_matrices(sys_par.flat_graph,
                                              conn_key=:learning_rule,
                                              pred=(x) -> !(x isa NoLearningRule),
                                              names_partitioned=sys_par.names_partitioned,
                                              subsystems_partitioned=sys_par.subsystems_partitioned).connection_matrices
    conn = prob.p.connection_matrices
    GDYAgent(sys_par, prob, policy, learning_rules, conn)
end

function run_experiment!(agent::GDYAgent, env::ClassificationEnvironment, save_path::Union{Nothing, String}=nothing, blocks=nothing;
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

function run_warmup(agent::GDYAgent, env::ClassificationEnvironment, t_warmup; alg, kwargs...)
    prob = remake(agent.problem; tspan=(0, t_warmup))
    sol = solve(prob, alg; save_everystep=false, kwargs...)
    u0 = sol[:,end] # last value of state vector
    return u0
end


function run_trial!(agent::GDYAgent, env; alg, kwargs...)
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
