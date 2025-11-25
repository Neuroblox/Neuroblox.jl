"""
    @graph

Helper macro that returns a directed graph whose nodes are Blox and whose edges are connections. This graph can then be turned into a system to be solved by calling `system_with_graph`.
Nodes should be declared within the block starting with @nodes, and connections should be declared within the block starting with @connections. Connections should be declared as Pairs, with an optional vector of keyword arguments coming after.

Example:
```julia
cortical = @graph begin
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
macro graph(body::Expr)
    output = Expr(:block)
    push!(output.args, :(g = $GraphDynamics.GraphSystem()))

    nodelist = Tuple{Symbol, Expr}[]
    nodelinenums = LineNumberNode[]
    edgelist = Tuple{Expr, Any}[]
    edgelinenums = LineNumberNode[]

    for expr in body.args
        if expr isa Expr && expr.head == :macrocall
            if expr.args[1] === Symbol("@nodes")
                parse_nodes!(nodelist, nodelinenums, expr.args[end])
            elseif expr.args[1] === Symbol("@connections")
                parse_edges!(edgelist, edgelinenums, expr.args[end])
            else
                error("Unknown macro call $(expr.args[1]) found in the @graph macro. Use @nodes to define nodes and @connections to define edges.")
            end
        else
            push!(output.args, expr)
        end
    end

    for (l, (lhs, rhs)) in zip(nodelinenums, nodelist)
        push!(output.args, l)
        push!(output.args, :($ModelingToolkit.@named $lhs = $rhs))
        push!(output.args, :($GraphDynamics.add_node!(g, $lhs)))
    end
    for (l, (e, kwargs)) in zip(edgelinenums, edgelist)
        push!(output.args, l)
        push!(output.args, :($GraphDynamics.add_connection!(g, $e[1], $e[2]; $(kwargs...))))
    end
    push!(output.args, :(g))
    esc(output)
end

macro graph()
    :(g = $GraphDynamics.GraphSystem())
end

function parse_nodes!(nodelist, nodelinenums, body)
    if body.head === :block
        for expr in body.args
            parse_node!(nodelist, nodelinenums, expr)
        end
    else
        parse_node!(nodelist, nodelinenums, body)
    end
end

function parse_node!(nodelist, nodelinenums, expr)
    if expr isa LineNumberNode
        push!(nodelinenums, expr)
    elseif expr.head === :(=)
        push!(nodelist, (expr.args[1], expr.args[2]))
    else
        error("Malformed node definition $expr. Node definitions should be of the format `(var_name) =Blox(...)`")
    end
end

function parse_edges!(edgelist, edgelinenums, body)
    if body.head === :block
        for expr in body.args
            parse_edge!(edgelist, edgelinenums, expr)
        end
    else
        parse_edge!(edgelist, edgelinenums, body)
    end
end

function parse_edge!(edgelist, edgelinenums, expr)
    if expr isa LineNumberNode
        push!(edgelinenums, expr)
    elseif expr.head === :call && expr.args[1] === :(=>)
        push!(edgelist, (expr, ()))
    elseif expr.head === :tuple && length(expr.args) == 2
        kwargs = []
        for arg in expr.args[2].args
            (arg.head != :(=)) && error("Malformed keyword argument $arg.")
            (arg.args[1] isa Symbol) || error("Invalid keyword argument name $(arg.args[1]).")
            push!(kwargs, Expr(:kw, (arg.args[1]), arg.args[2]))
        end
        push!(edgelist, (expr.args[1], kwargs))
    else
        error("Malformed edge line $expr. The line should contain a pair x => y indicating which nodes are connected, followed by an optional vector of keyword arguments separated by a comma.")
    end
end
