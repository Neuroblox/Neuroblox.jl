function find_blox(g::MetaDiGraph, blox)
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        b == blox && return v
    end

    return nothing
end

has_blox(g::MetaDiGraph, blox) = isnothing(find_blox(g, blox)) ? false : true

function add_edge!(g::MetaDiGraph, p::Pair; kwargs...)
    src, dest = p
    
    src_idx = find_blox(g, src)
    
    if isnothing(src_idx)
        add_blox!(g, src)
        src_idx = nv(g)
    end
    
    dest_idx = find_blox(g, dest)

    if isnothing(dest_idx)
        add_blox!(g, dest)
        dest_idx = nv(g)
    end

    add_edge!(g, src_idx, dest_idx, Dict(kwargs))
end

function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

function get_bloxs(g::MetaDiGraph)
    bs = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if !(b isa AbstractActionSelection)
            push!(bs, b)
        end
    end

    return bs
end

get_system(g::MetaDiGraph) = get_system.(get_bloxs(g))

get_dynamics_bloxs(blox) = [blox]
get_dynamics_bloxs(blox::CompositeBlox) = get_parts(blox)

flatten_graph(g::MetaDiGraph) = mapreduce(get_dynamics_bloxs, vcat, get_bloxs(g))

function connectors_from_graph(g::MetaDiGraph)
    conns = reduce(vcat, get_connectors.(get_bloxs(g)))
    for edge in edges(g)

        blox_src = get_prop(g, edge.src, :blox)
        blox_dest = get_prop(g, edge.dst, :blox)

        kwargs = props(g, edge)
        push!(conns, Connector(blox_src, blox_dest; kwargs...))
    end
   
    filter!(conn -> !isempty(conn), conns)

    return conns
end

function connector_from_graph(g::MetaDiGraph)
    conns = connectors_from_graph(g)

    return isempty(conns) ? Connector(:none, :none) : reduce(merge!, conns)
end

# Helper function to get delays from a graph
function graph_delays(g::MetaDiGraph)
    conn = connector_from_graph(g)

    return conn.delay
end

generate_discrete_callbacks(blox, ::Connector; t_block = missing) = []

function generate_discrete_callbacks(blox::Union{LIFExciNeuron, LIFInhNeuron}, bc::Connector; t_block = missing)
    sa = spike_affects(bc)
    name_blox = namespaced_nameof(blox)
    sys = get_namespaced_sys(blox)

    states_affect, params_affect = get(sa, name_blox, (Num[], Num[]))

    # HACK : MTK will complain if the parameter vector passed to a functional affect
    # contains non-unique parameters. Here we sometimes need to pass duplicate parameters that 
    # affect states in the loop in LIF_spike_affect! .
    # Passing parameters with Symbol aliases bypasses this issue and allows for duplicates. 
    affect_pairs = if unique(params_affect) == length(params_affect)
        [p => Symbol(p) for p in params_affect]
    else
        map(params_affect) do p
            if count(pi -> Symbol(pi) == Symbol(p), params_affect) > 1
                p => Symbol(p, "_$(rand(1:1000))")
            else
                p => Symbol(p)
            end
        end
    end
    
    ps = vcat([
        sys.V_reset => Symbol(sys.V_reset), 
        sys.t_refract_duration => Symbol(sys.t_refract_duration), 
        sys.t_refract_end => Symbol(sys.t_refract_end), 
        sys.is_refractory => Symbol(sys.is_refractory)
    ], affect_pairs)
    
    cb = (sys.V > sys.θ) => (
        LIF_spike_affect!, 
        vcat(sys.V, states_affect), 
        ps, 
        [], 
        nothing
    )

    return cb
end

function generate_discrete_callbacks(blox::HHNeuronExciBlox, ::Connector; t_block = missing)
    if !ismissing(t_block)
        nn = get_namespaced_sys(blox)
        eq = nn.spikes_window ~ 0
        cb_spike_reset = (t_block + sqrt(eps(float(t_block)))) => [eq]
        
        return cb_spike_reset
    else
        return []
    end
end

function generate_discrete_callbacks(bc::Connector, eqs::AbstractVector{<:Equation}; t_block = missing)
    eqs_params = get_equations_with_parameter_lhs(eqs)
    dc = discrete_callbacks(bc)

    if !ismissing(t_block) && !isempty(eqs_params)
        cb_params = (t_block - sqrt(eps(float(t_block)))) => eqs_params
        return vcat(cb_params, dc)
    else
        return dc
    end 
end

function generate_discrete_callbacks(g::MetaDiGraph, bc::Connector, eqs::AbstractVector{<:Equation}; t_block = missing)
    bloxs = flatten_graph(g)

    cbs = mapreduce(vcat, bloxs) do blox
        generate_discrete_callbacks(blox, bc; t_block)
    end
    
    cbs_connections = generate_discrete_callbacks(bc, eqs; t_block)

    return vcat(cbs, cbs_connections)
end


"""
    system_from_graph(g::MetaDiGraph, p=Num[]; name, simplify=true, graphdynamics=false, kwargs...)

Take in a `MetaDiGraph` `g` describing a network of neural structures (and optionally a vector of extra parameters `p`) and construct a `System` which can be used to construct various `Problem` types (i.e. `ODEProblem`) for use with DifferentialEquations.jl solvers.

If `simplify` is set to `true` (the default), then the resulting system will have `structural_simplify` called on it with any remaining keyword arguments forwared to `structural_simplify`. That is,
```
@named sys = system_from_graph(g; kwarg1=x, kwarg2=y)
```
is equivalent to
```
@named sys = system_from_graph(g; simplify=false)
sys = structural_simplify(sys; kwarg1=x, kwarg2=y)
```
See the docstring for `structural_simplify` for information on which options it supports.

If `graphdynamics=true` (defaults to `false`), the output will be a `GraphSystem` from [GraphDynamics.jl](https://github.com/Neuroblox/GraphDynamics.jl), and the `kwargs` will be sent to the `GraphDynamics` constructor instead of using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl/). The GraphDynamics.jl backend is typically significantly faster for large neural systems than the default backend, but is experimental and does not yet support all Neuroblox.jl features. 
"""
function system_from_graph(g::MetaDiGraph, p::Vector{Num}=Num[]; name=nothing, t_block=missing, simplify=true, graphdynamics=false, kwargs...)
    if graphdynamics
        isempty(p) || error(ArgumentError("The GraphDynamics.jl backend does yet support extra parameter lists. Got $p."))
        GraphDynamicsInterop.graphsystem_from_graph(g; kwargs...)
    else
        if isnothing(name)
            throw(UndefKeywordError(:name))
        end
        
        conns = connectors_from_graph(g)
    
        return system_from_graph(g, conns, p; name, t_block, simplify, kwargs...)
    end
end

function system_from_graph(g::MetaDiGraph, conns::AbstractVector{<:Connector}, p::Vector{Num}=Num[]; name=nothing, t_block=missing, simplify=true, graphdynamics=false, kwargs...)
    bloxs = get_bloxs(g)
    blox_syss = get_system.(bloxs)

    bc = isempty(conns) ? Connector(name, name) : reduce(merge!, conns)

    eqs = equations(bc)
    eqs_init = mapreduce(get_input_equations, vcat, bloxs)
    accumulate_equations!(eqs_init, eqs)

    connection_eqs = get_equations_with_state_lhs(eqs_init)

    discrete_cbs = identity.(generate_discrete_callbacks(g, bc, eqs_init; t_block))

    sys = compose(System(connection_eqs, t, [], vcat(params(bc), p); name, discrete_events = discrete_cbs), blox_syss)
    if simplify
        structural_simplify(sys; kwargs...)
    else
        sys
    end
end

function system_from_parts(parts::AbstractVector; name)
    return compose(System(Equation[], t; name), get_system.(parts))
end

function action_selection_from_graph(g::MetaDiGraph)
    idxs = findall(vertices(g)) do v
        b = get_prop(g, v, :blox)
        b isa AbstractActionSelection
    end

    if isempty(idxs)
        #error("No action selection block was detected in the current model.")
        return nothing
    else
        if length(idxs) > 1
            error("Multiple action selection blocks are detected. Only one must be used in an experiment.")
        else
            idx = only(idxs)
            b = get_prop(g, idx, :blox)
            idx_neighbors = inneighbors(g, idx)
            @assert length(idx_neighbors) == 2 "Two blocks need to connect to the action selection $(nameof(b)) block"

            bns = get_prop.(Ref(g), idx_neighbors, :blox)
            connect_action_selection!(b, bns...)

            return b
        end
    end
end

function learning_rules_from_graph(g::MetaDiGraph)
    d = Dict(Num, AbstractLearningRule)()

    for v in vertices(g)
        b = get_prop(g, v, :blox)
        for vn in inneighbors(g, v)
            if has_prop(g, vn, v, :learning_rule)
                bn = get_prop(g, v, :blox)
                weight = get_prop(g, vn, v, :weight)
                w = generate_weight_param(bn, b; weight)
                d[w] = get_prop(g, vn, v, :learning_rule)
            end
        end
    end

    return d
end

