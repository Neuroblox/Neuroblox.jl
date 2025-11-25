struct Connector
    source::Vector{Symbol}
    destination::Vector{Symbol}
    equation::Vector{Equation}
    weight::Vector{Num}
    delay::Vector{Num}
    discrete_callbacks
    spike_affects::Dict{Symbol, Vector{Union{Tuple{Num, Num}, Equation}}}
    learning_rule::Dict{Num, AbstractLearningRule}
end

function Connector(
    src::Union{Symbol, Vector{Symbol}}, 
    dest::Union{Symbol, Vector{Symbol}}; 
    equation=Equation[], 
    weight=Num[], 
    delay=Num[], 
    discrete_callbacks=[], 
    spike_affects=Dict{Symbol, Vector{Tuple{Num, Num}}}(),
    learning_rule=Dict{Num, AbstractLearningRule}(),
    connection_blox=Set([])
    )
    filter!(x -> !isempty(last(x)), spike_affects)
    # Check if all weigths have NoLearningRule and if so don't keep them in the final Dict.
    U = narrowtype_union(learning_rule)
    learning_rule = U <: NoLearningRule ? Dict{Num, NoLearningRule}() : learning_rule

    Connector(
        to_vector(src), 
        to_vector(dest), 
        to_vector(equation), 
        to_vector(weight), 
        to_vector(delay), 
        to_vector(discrete_callbacks), 
        spike_affects, 
        learning_rule
    )
end

function Base.isempty(conn::Connector)
    return isempty(conn.equation) && isempty(conn.weight) && isempty(conn.delay) && isempty(conn.discrete_callbacks) && isempty(conn.spike_affects) && isempty(conn.learning_rule)
end

Base.show(io::IO, c::Connector) = print(io, "$(c.source) => $(c.destination) with ", c.equation)

function show_field(io::IO, v::AbstractVector, title)
    if !isempty(v)
        println(io, title, " :")
        for val in v
            println(io, "\t $(val)")
        end
    end
end

function show_field(io::IO, d::Dict, title)
    if !isempty(d)
        println(io, title, " :")
        for (k, v) in d
            println(io, "\t ", k, " => ", v)
        end
    end
end

show_spike_affect(io::IO, t::Tuple) = println(io, "\t $(first(t)) += $(last(t))")

show_spike_affect(io::IO, eq::Equation) = println(io, "\t $eq")

function Base.show(io::IO, ::MIME"text/plain", c::Connector)
    
    println(io, "Connections :")
    for (s, d) in zip(c.source, c.destination)
        println(io, "\t $(s) => $(d)")
    end

    show_field(io, c.equation, "Equations")
    show_field(io, c.weight, "Weights")
    show_field(io, c.delay, "Delays")

    d = Dict()
    for w in c.weight  
        if haskey(c.learning_rule, w)
            d[w] = c.learning_rule[w]
        end
    end
    show_field(io, d, "Plasticity model")

    for s in c.source
        if haskey(c.spike_affects, s)
            println(io, "$(s) spikes affect :")
            sa = c.spike_affects[s]
            for x in sa
               show_spike_affect(io, x)
            end
        end
    end
end

function accumulate_equations!(eqs::AbstractVector{<:Equation}, bloxs)
    init_eqs = mapreduce(get_input_equations, vcat, bloxs)
    accumulate_equations!(eqs, init_eqs)

    return eqs
end

function accumulate_equations!(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs1, eq)
        else
            eqs1[idx] = eqs1[idx].lhs ~ eqs1[idx].rhs + eq.rhs
        end
    end

    return eqs1
end

function accumulate_equations(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    eqs = copy(eqs1)
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs, eq)
        else
            eqs[idx] = eqs[idx].lhs ~ eqs[idx].rhs + eq.rhs
        end
    end

    return eqs
end

ModelingToolkit.equations(c::Connector) = c.equation

"""
    discrete_callbacks(c::Connector)

Get the discrete events of a connection. These include the affects triggered whenever a spike occurs, as well as other things (e.g. switch behavior at some time `t_event`).
"""
discrete_callbacks(c::Connector) = c.discrete_callbacks

"""
    sources(c::Connector)

Get the source Blox of the connection.
"""
sources(c::Connector) = c.source

"""
    destinations(c::Connector)

Get the destination Blox of the connection.
"""
destinations(c::Connector) = c.destination

"""
    weights(c::Connector)

Get the weight or weight matrix for a connection.
"""
weights(c::Connector) = c.weight

"""
    delays(c::Connector)

Get the delays of the connection.
"""
delays(c::Connector) = c.delay

"""
    spike_affects(c::Connector)

Get the affects that are triggered every time the pre-synaptic Blox fires.
"""
spike_affects(c::Connector) = c.spike_affects

"""
    learning_rules(c::Connector)

Get the learning rule for the weight of the connector. Examples are [`NeurobloxPharma.HebbianPlasticity`](@ref) and [`NeurobloxPharma.HebbianModulationPlasticity`](@ref).
"""
learning_rules(c::Connector) = c.learning_rule

learning_rules(conns::AbstractVector{<:Connector}) = mapreduce(c -> c.learning_rule, merge!, conns)

get_equations_with_parameter_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> isparameter(eq.lhs), eqs)

get_equations_with_state_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> !isparameter(eq.lhs), eqs)

"""
    get_states_spikes_affect(sa, name)

Return the set of affects modifying the state of a Blox that should be triggered 
on an action potential.
"""
function get_states_spikes_affect(sa, name) 
    if haskey(sa, name)
        return first.(sa[name])
    else
        Num[]
    end
end

"""
    get_params_spikes_affect(sa, name)

Return the set of affects modifying the parameters of a Blox that should be 
triggered on an action potential.
"""
function get_params_spikes_affect(sa, name) 
    if haskey(sa, name)
        return last.(sa[name])
    else
        Num[]
    end
end

"""
    generate_weight_param(blox_out, blox_in; kwargs...)

Define a ModelingToolkit parameter corresponding to the connection weight between 
two Blox, or return such a parameter if it already exists.
"""
function generate_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    weight = get_weight(kwargs, name_out, name_in)
    if typeof(weight) == Num   # Symbol
        w = weight
    else
        w_name = Symbol("w_$(name_out)_$(name_in)")
        w = only(@parameters $(w_name)=weight [tunable=false])
    end    

    return w
end

"""
    generate_gap_weight_param(blox_out, blox_in; kwargs...)

Define a ModelingToolkit parameter corresponding to the gap junction weight between 
two Blox, or return such a parameter if it already exists.
"""
function generate_gap_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    gap_weight = get_gap_weight(kwargs, name_out, name_in)
    gw_name = Symbol("g_w_$(name_out)_$(name_in)")
    if typeof(gap_weight) == Num   # Symbol
        gw = gap_weight
    else
        gw = only(@parameters $(gw_name)=gap_weight)
    end    

    return gw
end

"""
    params(bc::Connector)

Helper to merge delay and weight of a connector into a single vector, as 
well as extract parameters from weights defined using expressions.
"""
function params(bc::Connector)
    wt = map(weights(bc)) do w
        Symbolics.get_variables(w)
    end

    if isempty(wt)
        return vcat(wt, delays(bc))
    else
        return vcat(reduce(vcat, wt), delays(bc))
    end
end

function Base.merge!(c1::Connector, c2::Connector)
    append!(c1.source, c2.source)
    append!(c1.destination, c2.destination)
    accumulate_equations!(c1.equation, c2.equation)
    append!(c1.weight, c2.weight)
    append!(c1.delay, c2.delay)
    append!(c1.discrete_callbacks, c2.discrete_callbacks)
    mergewith!(append!, c1.spike_affects, c2.spike_affects)
    merge!(c1.learning_rule, c2.learning_rule)
    return c1
end

Base.merge(c1::Connector, c2::Connector) = Base.merge!(deepcopy(c1), c2)

"""
    hypergeometric_connections(neurons_src, neurons_dst)

Create connections between two populations of neurons. For each postsynaptic 
(destination) neuron, randomly generate connections from the pool of presynaptic 
neurons, while guaranteeing that almost all source neurons have the same number 
of connections, and almost all destination neurons have the same number of 
connections.

Keyword arguments: 
    - `rng`: choice of random number generator
    - `density`: specifies the total number of connections (with density = 1 being fully connected)
    - `weight`: a number or vector indicating the weights of each connection
"""
function hypergeometric_connections(neurons_src, neurons_dest, name_out, name_in; rng=default_rng(), kwargs...)
    density = get_density(kwargs, name_out, name_in)
    N_connects =  density * length(neurons_dest) * length(neurons_src)
    out_degree = Int(ceil(N_connects / length(neurons_src)))
    in_degree =  Int(ceil(N_connects / length(neurons_dest)))
    wt = get_weight(kwargs,name_out, name_in)
    C = Connector[]
    outgoing_connections = zeros(Int, length(neurons_src))
    for neuron_postsyn in neurons_dest
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rng, rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_src[idx]
                push!(C, Connector(neuron_presyn, neuron_postsyn; kwargs...))
            end
        else
            for i in idx 
                kwargs = (kwargs...,weight=wt[i])
                push!(C, Connector(neurons_src[i], neuron_postsyn; kwargs...))
            end
        end
        outgoing_connections[idx] .+= 1
    end

    return reduce(merge!, C)
end

function indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    rng = get(kwargs, :rng, default_rng())
    in_degree =  Int(ceil(density * N_src))
    conn_mat = falses(N_src, N_dst)
    for j ∈ 1:N_dst
        idx = sample(rng, 1:N_src, in_degree; replace=false)
        for i ∈ idx
            conn_mat[i, j] = true
        end
    end
    conn_mat
end

"""
    indegree_constrained_connections(neurons_src, neurons_dst, name_src, name_dst; 
                                     kwargs...)
    
Create connections between two populations of neurons such that every destination 
neuron has the same in-degree, but no guarantees are made for the degrees of 
source neurons.
    
Keyword arguments: 
- `connection_matrix`: pre-specify the connection matrix.
"""
function indegree_constrained_connections(neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = get_density(kwargs, name_src, name_dst)
        indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end

    C = Connector[]
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                push!(C, Connector(neurons_src[i], neurons_dst[j]; kwargs...))
            end
        end
    end

    return reduce(merge!, C)
end

"""
    density_connections(neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    
Create connections between two populations of neurons. Unlike 
`hypergeometric_connections`, this process is entirely random and will not 
guarantee that neurons have roughly equal numbers of connections.
    
Keyword arguments: 
    - `rng`: choice of random number generator
    - `density`: specifies the total number of connections (with density = 1 being fully connected)
    - `weight`: a number or vector indicating the weights of each connection
"""
function density_connections(neurons_src, neurons_dst, name_src, name_dst; rng = default_rng(), kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_dst = length(neurons_dst)

    C = Connector[]
    for ns in neurons_src
        idxs = findall(rand(rng, N_dst) .<= density)
        for i in idxs
            push!(C, Connector(ns, neurons_dst[i]; kwargs...))
        end
    end

    return reduce(merge!, C)
end

function weight_matrix_connections(neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get_weightmatrix(kwargs, name_src, name_dst)

    if size(conn_mat) != (N_src, N_dst)
        error("The connection matrix must be of size $(N_src) x $(N_dst)")
    end

    C = Connector[]
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if !iszero(conn_mat[i, j])
                kwargs_ij = (kwargs..., weight=conn_mat[i, j])
                push!(C, Connector(neurons_src[i], neurons_dst[j]; kwargs_ij...))
            end
        end
    end

    return reduce(merge!, C)
end

connection_rule(blox_src, blox_dest; kwargs...) = Connector(blox_src, blox_dest; kwargs...)

connection_equations(blox_src, blox_dest; kwargs...) = Connector(blox_src, blox_dest; kwargs...).equation

connection_equations(source, destination, w; kwargs...) = Equation[]

"""
    connection_equations(source, destination, w; kwargs...)

Return the list of equations for a connection. This should be implemented any 
time one creates a new type of connection between Blox. If the connection is 
event-based, implement `connection_callbacks` as well.
"""
function connection_equations(blox_src::AbstractNeuron, blox_dest::AbstractNeuron, w; kwargs...)
    cr = get_connection_rule(kwargs, blox_src, blox_dest, w)

    @warn "The default connection equation `jcn ~ $(cr)` is used."
    return blox_dest.jcn ~ cr
end

"""
    connection_spike_affects(source, destination, w)

Return the list of affects that occur every time the presynaptic neuron fires. This 
should be implemented any time one creates a new type of connection that uses event-based spiking between Blox.
"""
connection_spike_affects(source, destination, w) = Tuple{Num, Num}[]

"""
    connection_learning_rule(source, destination, w; kwargs...)

For a connection with weight `w`, return a dictionary mapping `w` to its learning rule,
which is passed in as a `learning_rule` kwarg to the `Connector` constructor.
"""
function connection_learning_rule(source, destination, w; kwargs...)
    if haskey(kwargs, :learning_rule)
        return Dict(w => deepcopy(kwargs[:learning_rule]))
    else
        return Dict{Num, AbstractLearningRule}()
    end
end
    
"""
    connection_callbacks(source, destination; kwargs...)

Returns the callbacks for the connection, if it is an event-based connection. This 
should be implemented any time one creates a new type of connection between Blox. 
If the connection is continuous, implement `connection_equations` as well.
"""
connection_callbacks(source, destination; kwargs...) = []

function Connector(blox_src::AbstractBlox, blox_dest::AbstractBlox; kwargs...)
    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = connection_equations(blox_src, blox_dest, w; kwargs...)
    lr = connection_learning_rule(blox_src, blox_dest, w; kwargs...)  
    cb = connection_callbacks(blox_src, blox_dest; kwargs...)

    affects_tuple = connection_spike_affects(blox_src, blox_dest, w)
    sa = Dict(namespaced_nameof(blox_src) => to_vector(affects_tuple))  
    
    return Connector(
        namespaced_nameof(blox_src), 
        namespaced_nameof(blox_dest);
        equation = eq, 
        weight = w,
        spike_affects = sa,
        discrete_callbacks = cb,
        learning_rule = lr
    )
end

function Connector(
    blox_src::AbstractNeuralMass, 
    blox_dest::AbstractNeuralMass; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))
    x = only(outputs(blox_src; namespaced=true))
    if haskey(kwargs, :delay)
        error("Delay connections are currently not supported")
        # delay = get_delay(kwargs, nameof(sys_src), nameof(sys_dest))
        # τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        # τ = only(@parameters $(τ_name)=delay)
        # eq = sys_dest.jcn ~ x(t-τ)*w
        # return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ, learning_rule=Dict(w => lr))
    else
        eq = sys_dest.jcn ~ x*w
        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, learning_rule=Dict(w => lr))
    end    
end

function Connector(
    blox_src::AbstractNeuralMass, 
    blox_dest::AbstractObserver;
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    x = only(outputs(blox_src; namespaced=true))
    if x isa Num
        w = generate_weight_param(blox_src, blox_dest; kwargs...)
        eq = sys_dest.jcn ~ x*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    else
        # Need t for the delay term
        @variables t
        # Define & accumulate delay parameter
        # Don't accumulate if zero
        τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        τ = only(@parameters $(τ_name)=delay)

        w_name = Symbol("w_$(nameof(sys_src))_$(nameof(sys_dest))")
        w = only(@parameters $(w_name)=weight)
        
        eq = sys_dest.jcn ~ x(t-τ)*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ)
    end
end

function Connector(
    blox_src::AbstractSimpleStimulus,
    blox_dest::Union{AbstractNeuralMass, AbstractNeuron};
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

Connector(blox::AbstractBlox, as::AbstractActionSelection; kwargs...) = Connector(namespaced_nameof(blox), namespaced_nameof(as))

function Connector(
    blox_src::AbstractNeuron, 
    blox_dest::AbstractNeuralMass; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(outputs(blox_src; namespaced=true))
    if x isa Num
        eq = sys_dest.jcn ~ x*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    else
        @variables t
        delay = get_delay(kwargs, nameof(blox_src), nameof(blox_dest))
        τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        τ = only(@parameters $(τ_name)=delay)

        eq = sys_dest.jcn ~ x(t-τ)*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ)
    end
end
