"""
    @connection function (conn::ConnType)(src::SrcType, dst::DstType, t)
        # assignments...
        
        @equations begin
            ...
        end

        # Optional: define discrete events
        @event_times

        @discrete_events begin 
            ...
        end
    end

Define a connection between two non-composite blox, using its equations and discrete events. Must specify the conneciton type and the types of the source and destination blox. Access fields of the blox or connections inside the macro using standard Julia syntax (e.g. src.V for the voltage of the source blox).

The `@equations` block is mandatory, and the `@discrete_events` and `@event_times` blocks are used if the connection has discrete events attached to it.
```
"""
macro connection(body::Expr)
    call_expr = body.args[1]
    fn_body = body.args[2]

    if body.head !== :function || call_expr.head !== :call
        error("Connection macro must take the form of a function definition, with type annotations for the type of connection, source blox, and destination blox.")
    elseif length(call_expr.args) !== 4
        error("Incorrect number of function arguments. Function must take 3 arguments: a source blox, a destination blox, and a time argument.")
    else
        var_names = Union{Symbol, Expr}[]
        var_types = Union{Symbol, Expr}[]
        for arg in call_expr.args[1:end-1]
            if arg.head !== :(::)
                error("Missing type annotation for $arg. The connection, source blox, and destination blox must all have type annotations.")
            else
                push!(var_names, arg.args[1])
                push!(var_types, arg.args[2])
            end
        end
        if call_expr.args[end] isa Symbol
            push!(var_names, call_expr.args[end])
        else
            push!(var_names, call_expr.args[end].args[1])
        end
        build_connection(var_types, var_names, fn_body)
    end
end

function build_connection(arg_types::Vector{Union{Symbol, Expr}}, arg_names::Vector{Union{Symbol, Expr}}, body::Expr)
    conn_type, src_type, dst_type = esc.(arg_types)
    output = Expr(:block)
    push!(output.args, :(src_type = $src_type))
    push!(output.args, :(conn_type = $conn_type))
    push!(output.args, :(dst_type = $dst_type))

    assignments = Expr(:block)

    for (lnn, arg) in zip(body.args, body.args[2:end])
        if !(lnn isa LineNumberNode) || !(arg isa Expr)
            continue
        end
        if arg.head === :(=)
            push!(assignments.args, lnn)
            push!(assignments.args, arg)
        elseif arg.head === :macrocall && arg.args[1] === Symbol("@equations")
            eqs_expr = parse_conn_equations(arg_types, arg_names, arg, assignments)
            push!(output.args, eqs_expr)
        elseif arg.head === :macrocall && arg.args[1] === Symbol("@discrete_events")
            events_expr = parse_conn_events(arg_types, arg_names, arg, assignments)
            push!(output.args, events_expr)
        elseif arg.head === :macrocall && arg.args[1] === Symbol("@event_times")
            evtimes_expr = parse_event_times(arg_types, arg_names, arg, assignments)
            push!(output.args, evtimes_expr)
        else
            error("Invalid line $arg in the @connection block.")
        end
    end

    output
end

function parse_conn_equations(arg_types, arg_names, body, assignments)
    eq_block = body.args[end]
    output_nt = :((;))
    eq_def = Expr(:block)
    linenums = LineNumberNode[]
    inputs_assigned = Expr(:tuple)

    for item in eq_block.args
        if item isa LineNumberNode
            push!(eq_def.args, item)
        elseif item isa Expr && item.head === :(=)
            esc_input_var = esc(item.args[1])
            push!(eq_def.args, :($esc_input_var = $(item.args[2])))
            push!(output_nt.args[1].args, :($esc_input_var = $esc_input_var))
            push!(inputs_assigned.args, QuoteNode(item.args[1]))
        else
            error("Invalid line $item in the @equations block. Each line of the equations block should be an equation specifying the value of a variable input to the destination blox.")
        end
    end
    push!(eq_def.args, :(out = $output_nt))

    conn, src, dst, t = arg_names
    quote
        function ($conn::conn_type)($src::GraphDynamics.Subsystem{<:src_type}, $dst::GraphDynamics.Subsystem{<:dst_type}, $t)
            acc = GraphDynamics.initialize_input(GraphDynamics.to_subsystem($dst))
            inputs = propertynames(acc)
            if any(∉(inputs), $inputs_assigned)
                nonmatching_input_error(inputs, $inputs_assigned, GraphDynamics.get_tag($dst))
            end
            $assignments
            $eq_def
            merge(acc, out)
        end
    end
end
@noinline function nonmatching_input_error(inputs, inputs_assigned, dst_type)
    nonmatched = filter(∉(inputs), inputs_assigned)
    throw(ArgumentError("Error applying connection rule. $nonmatched are not input symbols of $dst_type, valid inputs are $inputs"))
end

function parse_conn_events(arg_types, arg_names, body, assignments)
    de_block = body.args[end]
    conn, src, dst, t = esc.(arg_names)
    conn_type, src_type, dst_type = arg_types

    conds = Expr[]
    affects = Expr[]
    linenums = LineNumberNode[]
    for line in de_block.args
        if line isa LineNumberNode
            push!(linenums, line)
        elseif line.head == :call && line.args[1] == :(=>)
            push!(conds, line.args[2])
            push!(affects, line.args[3])
        else
            error("Invalid line $line in the @discrete_events block. Each line of the block should be a callback specified using the pair syntax.")
        end
    end

    callbacks = collect(zip(conds, affects))
    condition_code = generate_conditional(conds, fill(:(return true), length(conds)))
    cb_code = generate_callback_block(callbacks, linenums, arg_types, arg_names)

    quote
        GraphDynamics.has_discrete_events(::Type{conn_type}, ::Type{<:src_type}, ::Type{<:dst_type}) = true

        function GraphDynamics.discrete_event_condition($conn::conn_type, $t, $src, $dst) 
            $assignments
            $condition_code
            return false
        end

        function GraphDynamics.apply_discrete_event!($t, sysview_src, sysview_dst, $conn::conn_type, $src::GraphDynamics.Subsystem{<:src_type}, $dst::GraphDynamics.Subsystem{<:dst_type})
            # hack: name the integrator argument t so that the t in the condition is properly escaped
            $t = $t.t
            $assignments
            $cb_code
        end
    end
end

function parse_event_times(arg_types, arg_names, block, assignments)
    if length(block.args) != 3
        error("@event_times should only take a single argument. If a set of event times is desired, put an iterable containing the event times.")
    end
    output = block.args[end]
    conn, src, dst, t = esc.(arg_names)

    quote
        function GraphDynamics.event_times(conn::conn_type, ::GraphDynamics.Subsystem{<:src_type}, ::GraphDynamics.Subsystem{<:dst_type})
            $assignments
            $output
        end
    end
end

function generate_conditional(conds, bodies)
    @assert length(conds) == length(bodies)
    block = Expr(:block)
    for (cond, body) in zip(conds, bodies)
        push!(block.args, :(if $cond
                            $body 
                        end))
    end
    return block
end

function generate_callback_block(callbacks, linenums, arg_types, arg_names)
    contents = zip(callbacks, linenums)
    bodies = map(contents) do ((cond, aff), l)
        generate_affect_code(aff, l, arg_types, arg_names)
    end
    conds = first.(callbacks)
    generate_conditional(conds, bodies)
end

function generate_affect_code(aff_eqs, lnn, arg_types, arg_names)
    block = Expr(:block)
    push!(block.args, lnn)
    if aff_eqs.head == :(=)
        lhs, rhs = aff_eqs.args
        node, field = _src_or_dst(lhs, arg_types, arg_names)
        push!(block.args, :($node.$field[] = $rhs))
    elseif aff_eqs.head == :vect
        for eq in aff_eqs.args
            lhs, rhs = eq.args
            node, field = _src_or_dst(lhs, arg_types, arg_names)
            push!(block.args, :($node.$field[] = $rhs))
        end
    else
        error("Malformed set of affect equations $aff_eqs.")
    end
    return block
end

"""
Determine if a symbol like like src.H is a state or parameter
"""
function _src_or_dst(lhs, arg_types, arg_names)
    conn_type, src_type, dst_type = arg_types
    conn, src, dst, t = arg_names
    node, sym = lhs.args[1], lhs.args[2].value

    if node == dst
        :sysview_dst, sym
    elseif node == src
        :sysview_src, sym
    else
        error("Invalid expression in a discrete event affect $lhs: the name $node does not actually match the name of the source or destination argument.")
    end
end
