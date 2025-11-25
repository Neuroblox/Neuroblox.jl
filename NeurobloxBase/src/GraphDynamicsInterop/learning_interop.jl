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

function NeurobloxBase.action_selection_from_graph(g::GraphSystem)
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
