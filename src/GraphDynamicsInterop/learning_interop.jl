function maybe_set_state_pre(lr::AbstractLearningRule, state)
    if isnothing(lr.state_pre)
        @set lr.state_pre = state
    else
        lr
    end
end

function maybe_set_state_post(lr::AbstractLearningRule, state)
    if isnothing(lr.state_post)
        @set lr.state_post = state
    else
        lr
    end
end
maybe_set_state_pre(lr::NoLearningRule, state) = lr
maybe_set_state_post(lr::NoLearningRule, state) = lr

function Neuroblox.action_selection_from_graph(g::GraphSystem)
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

struct GDYAgent{S,P,A,LR,CM} <: Agent
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connection_matrices::CM
end
function Neuroblox.Agent(g::GraphSystem; name, t_block=missing, u0=[], p=[], graphdynamics=true, kwargs...)
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
                                              pred=(x) -> !(x isa NoLearningRule)).connection_matrices
    conn = prob.p.connection_matrices
    GDYAgent(sys_par, prob, policy, learning_rules, conn)
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

maybe_show_plot(args...; kwargs...) = []
showvalues(;trace, N_trials) = () -> begin
    pct_correct = round(sum(row -> row.iscorrect, trace)*100/length(trace); digits=3)
    len=min(100, length(trace))
    pct_correct_recent = let
        pct = sum(1:len) do i
            trace[end-i+1].iscorrect
        end / len
        (round(pct*100; digits=3))
    end
    N = 8
    last_N = map(min(length(trace)-1, N-1):-1:0) do i
        (;trial, iscorrect, action, time) = trace[end-i]
        Response = iscorrect ? "\e[0;32mCorrect\e[0m," : "\e[0;31mFalse\e[0m,  "
        trial_str = rpad("$trial,", textwidth(string(N_trials))+1)
        ("Trial", "$trial_str Category choice = $(action), Response = $Response Time = $(round(time, digits=3)) seconds")
    end
    [
        last_N
        ("Accuracy", "$(pct_correct)% total, $(pct_correct_recent)% last $len trials")
        maybe_show_plot(trace)
    ]
end

function runningperf(trace; len=min(150, length(trace)))
    sum(1:len) do i
        trace[end-i+1].iscorrect
    end / len
end

function Neuroblox.run_experiment!(agent::GDYAgent, env::ClassificationEnvironment; t_warmup=0, verbose=false, kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)
    @time begin
        if t_warmup > 0
            u0 = @noinline run_warmup(agent, env, t_warmup; kwargs...)
            @reset agent.problem = remake(agent.problem; tspan, u0=u0)
        else
            @reset agent.problem = remake(agent.problem; tspan)
        end
        print("Warmed up in: ")
    end
    trace = @NamedTuple{trial::Int, iscorrect::Bool, action::Int, time::Float64}[]
    prog = Progress(N_trials; showspeed=true, enabled=verbose)
    try
        for trial ∈ 1:N_trials
            (;time, gctime) = @timed begin
                _, iscorrect, action = @noinline run_trial!(agent, env; kwargs...)
            end
            push!(trace, (;trial, iscorrect, action, time))            
            next!(prog, showvalues=showvalues(;trace, N_trials))
        end
    catch e;
        if e isa InterruptException
            @warn "Interrupted! Bailing out now"
            finish!(prog)
        else
            rethrow(e)
        end
    end
    trace
end

function Neuroblox.run_warmup(agent::GDYAgent, env::ClassificationEnvironment, t_warmup; alg, kwargs...)
    prob = remake(agent.problem; tspan=(0, t_warmup))
    sol = solve(prob, alg; save_everystep=false, kwargs...)
    u0 = sol[:,end] # last value of state vector
    return u0
end


function Neuroblox.run_trial!(agent::GDYAgent, env; alg, kwargs...)
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

function apply_learning_rules!(sol, prob, learning_rules, feedback)
    (;connection_matrices, params_partitioned, names_partitioned) = prob.p
    _apply_learning_rules!(sol, params_partitioned, connection_matrices, learning_rules,
                           names_partitioned,
                           feedback)
end

function _apply_learning_rules!(sol,
                                params_partitioned::NTuple{Len, Any},
                                connection_matrices::ConnectionMatrices{NConn},
                                learning_rules::ConnectionMatrices{NLearn},
                                names_partitioned,
                                feedback) where {Len, NConn, NLearn}
    for i ∈ eachindex(params_partitioned)
        for k ∈ eachindex(params_partitioned)
            for ncl ∈ 1:length(learning_rules)
                M_learning = learning_rules[ncl].data[k][i]  
                if !(M_learning isa NotConnected)
                    for nc ∈ 1:length(connection_matrices)
                        M = connection_matrices[nc].data[k][i]
                        if !(M isa NotConnected)
                            for j ∈ eachindex(params_partitioned[i])
                                for (l, rule) ∈ maybe_sparse_enumerate_col(M_learning, j)
                                    if Base.isstored(M, l, j)
                                        conn = M[l, j]
                                        Δw = weight_gradient(rule, sol, conn.weight, feedback)
                                        name_dst = names_partitioned[i][j]
                                        name_src = names_partitioned[k][l]
                                        if !isfinite(Δw)
                                            @warn "non-finite gradient" name_dst name_src Δw
                                        end
                                        M[l, j] = @reset conn.weight += (Δw)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
