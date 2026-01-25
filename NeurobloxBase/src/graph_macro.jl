"""
    @graph

Helper macro that returns a GraphSystem whose nodes are Blox and whose edges are connections.
Nodes should be declared within the block starting with @nodes, and connections should be declared
within the block starting with @connections. Connections should be declared as Pairs, with an optional
vector of keyword arguments coming after.

Example:
```julia
cortical = @graph name begin
    @nodes begin
        ASC1 = NextGenerationEI(; ...)
        Layer_2_3_A = Cortical(; density = 0.03) 
    end

    @connections begin
        ASC1 => Layer_2_3_A, [weight = 20]
    end
end
```
"""
macro graph(namespace::Symbol, body::Expr)
    graph_macro(:($namespace = $GraphSystem(; name=$(QuoteNode(namespace)))), body)
end

macro graph(body::Expr)
    graph_macro(:($GraphSystem(; name=$nothing)), body)
end

macro graph()
    :(GraphSystem(; name=nothing))
end

"""
    @graph! g expr

Like the `@graph` macro, except it takes an existing graph `g` as its first argument and mutates it by adding
additional nodes and connections.
"""
macro graph!(g, body)
    graph_macro(g, body)
end

function graph_macro(g_input, body)
    @gensym g namespace
    output = Expr(:block,
                  :($g::$(Union{GraphSystem, PartitioningGraphSystem}) = $g_input),
                  :($namespace = $getfield($g, :name)),
                  )

    nodelist = Tuple{Symbol, Expr}[]
    nodelinenums = LineNumberNode[]
    edgelist = Tuple{Expr, Any}[]
    edgelinenums = LineNumberNode[]
    
    body = let
        sub_nodes = Substitute() do expr
            isexpr(expr, :macrocall) && expr.args[1] == Symbol("@nodes")
        end
        sub_nodes(body) do expr
            expr_block = Expr(:block, expr.args[2:end]...)
            substitute_nodes(expr_block, g, namespace)
        end
    end
    body = let
        sub_connections = Substitute() do expr
            isexpr(expr, :macrocall) && expr.args[1] == Symbol("@connections")
        end
        sub_connections(body) do expr
            expr_block = Expr(:block, expr.args[2:end]...)
            substitute_edges(expr_block, g, namespace)
        end
    end
    push!(output.args, body, g)
    return esc(output)
end

function substitute_nodes(nodeblock, g, namespace)
    function walk_and_replace(ex)
        if isexpr(ex, :macrocall, 3) && ex.args[1] == Symbol("@rule") && isexpr(ex.args[3], :(=))
            # Turn `@rule x = y` into system_wiring_rule!(g, x)
            _substitude_node(ex.args[3], g, namespace, system_wiring_rule!)
        elseif isexpr(ex, :(=))
            # Turn `x = y` into add_node!(g, x)
            _substitude_node(ex, g, namespace, add_node!)
        elseif ex isa Expr
            Expr(ex.head, walk_and_replace.(ex.args)...)
        else
            ex
        end
    end
    walk_and_replace(nodeblock)
end

function _substitude_node(expr, g, namespace, func!)
    @gensym idx res
    @match expr begin
        :($a = [$f($(args...)) for $c ∈ $d]) => begin
            call = inferred_name_namespace_fexpr(f, args, a, namespace, idx)
            body = Expr(:block,
                        :($res = $call),
                        :($func!($g, $res)),
                        res)
            :($a = [$body for ($idx, $c) ∈ enumerate($d)])
        end
        :($a = for $c ∈ $d
              $(assignments...)
              $f($(args...))
              $lnn
          end) => begin
              call = inferred_name_namespace_fexpr(f, args, a, namespace, idx)
              body = Expr(:block,
                          assignments...,
                          :($res = $call),
                          :($func!($g, $res)),
                          res)
              :($a = [$body for ($idx, $c) ∈ enumerate($d)])
          end
        :($a = $f($(args...))) => begin
            call = inferred_name_namespace_fexpr(f, args, a, namespace, nothing)
            :($a = $call; $func!($g, $a); $a)
        end
        ex => error("Unrecognized syntax in @nodes block:\n $ex")
    end
end

function inferred_name_namespace_fexpr(fname, fargs, name::Symbol, namespace, idxsym)
    args, kwargs = @match fargs begin
        [Expr(:parameters, kwargs...), args...] => handle_kwargs(args, kwargs, namespace, name, idxsym)
        [args...] => handle_kwargs(args, (), namespace, name, idxsym)
        ex => error("Invalid expression $(:($fname($(fargs...))))")
    end
    :($fname($(args...); $(kwargs...)))
end

function handle_kwargs(args_in, kwargs_in, namespace, name, idxsym)
    name_ex = isnothing(idxsym) ? QuoteNode(name) : :($Symbol($(QuoteNode(name)), $idxsym))
    kwargs = Any[Expr(:kw, :name, name_ex), Expr(:kw, :namespace, namespace)]
    args = []
    for arg in args_in
        if isexpr(arg, :kw)
            if arg.args[1] == :name
                kwargs[1] = arg
            elseif arg.args[1] == :namespace
                kwargs[2] = arg
            else
                push!(kwargs, arg)
            end
        else
            push!(args, arg)
        end
    end
    for kwarg in kwargs_in
        if isexpr(kwarg, :kw)
            if kwarg.args[1] == :name
                kwargs[1] = kwarg
            elseif kwarg.args[1] == :namespace
                kwargs[2] = kwarg
            else
                push!(kwargs, kwarg)
            end
        else
            push!(kwargs, kwarg)
        end
    end
    args, kwargs
end

function substitute_edges(edgeblock, g, namespace)
    function walk_and_replace(ex)
        # Turn `@rule x => y` into system_wiring_rule!(g, x)
        if isexpr(ex, :macrocall, 3) && ex.args[1] == Symbol("@rule")
            @match ex.args[3] begin
                :($a => $b)        => _substitude_edge(ex.args[3], g, namespace, system_wiring_rule!)
                :(($a => $b), $kw) => _substitude_edge(ex.args[3], g, namespace, system_wiring_rule!)
                ex => error("Invalid use of @rule, expected expression of the form @rule a => b, [kwargs...], got\n @rule $ex")
            end
        else
            @match ex begin
                :($a => $b)        => _substitude_edge(ex, g, namespace, add_connection!)
                :(($a => $b), $kw) => _substitude_edge(ex, g, namespace, add_connection!)
                ex::Expr           => Expr(ex.head, walk_and_replace.(ex.args)...)
                x                  => x
            end
        end
    end
    walk_and_replace(edgeblock)
end

function _substitude_edge(expr, g, namespace, func!)
    @match expr begin
        ::Symbol => expr
        :($src => $dst) => :($func!($g, $src, $dst))
        :($src => $dst, $kw) => begin
            kwargs = @match kw begin
                :([$(kwargs...)]) => make_kwarg.(kwargs)
                :(($(kwargs...),)) => make_kwarg.(kwargs)
                :((; $(kwargs...))) => make_kwarg.(kwargs)
            end
            :($func!($g, $src, $dst; $(kwargs...)))
        end
        ex => ex
    end
end

make_kwarg(kwarg) = @match kwarg begin
    s::Symbol => s
    :($name = $val) => Expr(:kw, name, val)
    Expr(:kw, name, val) => Expr(:kw, name, val)
    Expr(:(...), args...) => Expr(:(...), args...)
    ex => error("Malformed keyword argument $kwarg.")
end
