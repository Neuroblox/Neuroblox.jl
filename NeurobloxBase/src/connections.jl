"""
    get_weight(::ConnectionRule)

Returns the `weight` corresponding to a connection rule, if that connection rule corresponds to
a `weight` keyword arugment during graph creation, otherwise returns `nothing`.

E.g. this returns `nothing` on `ReverseConnection`s because the weights in the reverse connection
are just a mirror of a pre-existing `BasicConnection`.
"""
function get_weight end

##----------------------------------------------
"""
    BasicConnection{T}(weight)
    BasicConnection(weight::T)

The default connection type.
"""
Base.@kwdef struct BasicConnection{T} <: ConnectionRule
    weight::T
    BasicConnection{T}(x) where {T} = new{T}(x)
    BasicConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::BasicConnection, name_src, name_dst)
    (; weight = Symbol(:w_, name_src, :_, name_dst))
end
Base.zero(::Type{<:BasicConnection{T}}) where {T} = BasicConnection(zero(T))
Base.zero(::Type{BasicConnection}) = BasicConnection(0.0)

function (c::BasicConnection)(blox_src, blox_dst, t)
    (; jcn = c.weight * output(blox_src))
end

get_weight((; weight)::BasicConnection) = weight

##----------------------------------------------

Base.@kwdef struct ReverseConnection{T} <: ConnectionRule
    weight::T
    ReverseConnection{T}(x) where {T} = new{T}(x)
    ReverseConnection(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{ReverseConnection{T}}) where {T} = ReverseConnection(zero(T))
Base.zero(::Type{ReverseConnection}) = ReverseConnection(0.0)
get_weight(::ReverseConnection) = nothing

#----------------------

"""
    EventConnection{T}(weight, event_times::NamedTuple)
    EventConnection(weight::T, event_times::NamedTuple)

A connection containing a weight and a `NamedTuple` of labelled event times, often useful for connection-based events.
"""
Base.@kwdef struct EventConnection{T, NT <: NamedTuple} <: ConnectionRule
    weight::T
    event_times::NT
    EventConnection(w::T, event_times::NT) where {T, NT} = new{float(T), NT}(w, event_times)
    EventConnection{T}(w, event_times::NT) where {T, NT} = new{T, NT}(w, event_times)
end
Base.zero(::Type{<:EventConnection{T}}) where {T} = EventConnection(zero(T), (;))
Base.zero(::Type{<:EventConnection}) = EventConnection(0.0, (;))

GraphDynamics.has_discrete_events(::Type{<:EventConnection}, ::Type, ::Type) = true
function GraphDynamics.discrete_event_condition((;event_times)::EventConnection, t, src, dst)
    t ∈ event_times
end
GraphDynamics.event_times((;event_times)::EventConnection, src, dst) = event_times
get_weight((; weight)::EventConnection) = weight

##----------------------------------------------

Base.@kwdef struct PSPConnection{T} <: ConnectionRule
    weight::T
    PSPConnection{T}(x) where {T} = new{T}(x)
    PSPConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::PSPConnection, name_src, name_dst)
    (; weight = Symbol(:w_PSP_, name_src, :_, name_dst))
end
Base.zero(::Type{PSPConnection}) = PSPConnection(0.0)
Base.zero(::Type{PSPConnection{T}}) where {T} = PSPConnection(zero(T))

function (c::PSPConnection)(sys_src::GraphDynamics.Subsystem{<:AbstractNeuron}, sys_dst::GraphDynamics.Subsystem{<:AbstractNeuron}, t)
    (;jcn = c.weight * sys_src.G * (sys_src.E_syn - sys_dst.V))
end
get_weight((; weight)::PSPConnection) = weight

struct HHConnection_GAP{T} <: ConnectionRule
    w_gap::T
    HHConnection_GAP{T}(x) where {T} = new{T}(x)
    HHConnection_GAP(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP}) = HHConnection_GAP(0.0)
Base.zero(::Type{HHConnection_GAP{T}}) where {T} = HHConnection_GAP(zero(T))

function GraphDynamics.connection_property_namemap(::HHConnection_GAP, name_src, name_dst)
    (; w_gap = Symbol(:w_GAP_, name_src, :_, name_dst))
end
get_weight(::HHConnection_GAP) = nothing

##----------------------------------------------

Base.@kwdef struct HHConnection_GAP_Reverse{T} <: ConnectionRule
    w_gap_rev::T
    HHConnection_GAP_Reverse{T}(x) where {T} = new{T}(x)
    HHConnection_GAP_Reverse(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP_Reverse{T}}) where {T} = HHConnection_GAP_Reverse(zero(T))
Base.zero(::Type{HHConnection_GAP_Reverse}) = HHConnection_GAP_Reverse(0.0)

function GraphDynamics.connection_property_namemap(::HHConnection_GAP_Reverse, name_src, name_dst)
    (; w_gap_rev = Symbol(:w_GAP_reverse_, name_src, :_, name_dst))
end

get_weight(::HHConnection_GAP_Reverse) = nothing


#-----------------------------------------------

"""
    MultipointConnection(weight::T, member_indices::NamedTuple)

Construct a `MultipointConnection` that stores a named tuple of `PartitionedIndex`es for each
extra subsystem that might be involved in a given connection. This is useful for situations
where calculating the inputs to a given subsystem requires coordination between multiple
input subsystems.

The `PartitionedIndex` of a given blox in a `g::PartitioningGraphSystem` can be found via
`PartitionedIndex(g, node)`. 
"""
struct MultipointConnection{T, NT <: NamedTuple} <: ConnectionRule
    weight::T
    member_indices::NT
    function MultipointConnection(x, member_indices::NT) where {NT}
        fx = float(x)
        new{typeof(fx), NT}(fx, member_indices)
    end
end
@generated function Base.zero(::Type{MultipointConnection{T, NamedTuple{names, Idxs}}}) where {T, names, Idxs}
    # these indices are invalid and using them will error
    zinds = Tuple(map(zero, Idxs.parameters))
    ex = Expr(:tuple, (:(zero($Idx)) for Idx ∈ Idxs.parameters)...)
    :(MultipointConnection(zero(T), NamedTuple{names}($ex)))
end
Base.zero(::Type{MultipointConnection}) = MultipointConnection(0.0, (;))
Base.zero(::Type{MultipointConnection{T}}) where {T} = MultipointConnection(zero(T), (;))

GraphDynamics.connection_needs_ctx(::MultipointConnection) = true
function (conn::MultipointConnection)(sys_src::Subsystem, sys_dst::Subsystem, t::Real, ctx::NamedTuple)
    # Fetch all the intermediate objects
    intermediate_subsystems = map(conn.member_indices) do idx
        sys = Subsystem(ctx.states_partitioned[idx.i][idx.j], ctx.params_partitioned[idx.i][idx.j])
        sys
    end
    # Feed them back as arguments
    conn(sys_src, intermediate_subsystems..., sys_dst, t)
end
function GraphDynamics.connection_property_namemap(conn::MultipointConnection, name_src, name_dst)
    (; weight = Symbol(:w_, name_src, :_, join(keys(conn.member_indices), '_'), :_, name_dst))
end

function GraphDynamics.merge_connection!(g_new, g_old, src, conn::MultipointConnection, dst; kwargs...)
    # This gets called when merge!-ing a composite blox's graph into and outer graph. When that happens,
    # we need to update indices that originated in the old graph and make them apply to the new graph
    @reset conn.member_indices = map(conn.member_indices) do idx
        node = g_old.nodes_partitioned[idx]
        name = GraphDynamics.get_name(node)
        new_idx = g_new.node_namemap[name]
    end
    add_connection!(g_new, src, conn, dst; kwargs...)
end
get_weight((; weight)::MultipointConnection) = weight

"""
    hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)

Populate graph `g` with connections between two populations of neurons. For each postsynaptic 
(destination) neuron, randomly generate connections from the pool of presynaptic 
neurons, while guaranteeing that almost all source neurons have the same number 
of connections, and almost all destination neurons have the same number of 
connections.

Keyword arguments: 
    - `rng`: choice of random number generator
    - `density`: specifies the total number of connections (with density = 1 being fully connected)
    - `weight`: a number or vector indicating the weights of each connection
"""
function hypergeometric_connections!(g,
                                     neurons_src, neurons_dst, name_src, name_dst; rng=default_rng(), kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_connects = density * length(neurons_dst) * length(neurons_src)
    if length(neurons_src) == 0 || length(neurons_dst) == 0
        error(ArgumentError("hypergeometric_connections requires non-zero source and destination populations, got length(neurons_src) = $(length(neurons_src)), length(neurons_dst) = $(length(neurons_dst)) "))
    end
    out_degree = Int(ceil(N_connects / length(neurons_src)))
    in_degree =  Int(ceil(N_connects / length(neurons_dst)))
    wt = get_weight(kwargs, name_src, name_dst)
    # @info "" name_src name_dst (; density, out_degree, in_degree, N_connects)
    outgoing_connections = zeros(Int, length(neurons_src))
    for neuron_postsyn in neurons_dst
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rng, rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_src[idx]
                system_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        else
            for i in idx
                system_wiring_rule!(g, neurons_src[i], neuron_postsyn; kwargs..., weight=wt[i])
            end
        end
        outgoing_connections[idx] .+= 1
    end
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
    indegree_constrained_connections!(g, neurons_src, neurons_dst, name_src, name_dst; 
                                      kwargs...)
    
Populate graph `g` with connections between two populations of neurons such that every destination 
neuron has the same in-degree, but no guarantees are made for the degrees of 
source neurons.
    
Keyword arguments: 
- `connection_matrix`: pre-specify the connection matrix.
"""
function indegree_constrained_connections!(g,
                                           neurons_src, neurons_dst,
                                           name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = NeurobloxBase.get_density(kwargs, name_src, name_dst)
        NeurobloxBase.indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                system_wiring_rule!(g, neurons_src[i], neurons_dst[j]; kwargs...)
            end
        end
    end
end


"""
    density_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)

Populate graph `g` with connections between two populations of neurons. Unlike 
`hypergeometric_connections`, this process is entirely random and will not 
guarantee that neurons have roughly equal numbers of connections.
    
Keyword arguments: 
    - `rng`: choice of random number generator
    - `density`: specifies the total number of connections (with density = 1 being fully connected)
    - `weight`: a number or vector indicating the weights of each connection
"""
function density_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_dst = length(neurons_dst)
    rng = get(kwargs, :rng, default_rng())

    for ns in neurons_src
        idxs = findall(rand(rng, N_dst) .<= density)
        for i in idxs
            system_wiring_rule!(g, ns, neurons_dst[i]; kwargs...)
        end
    end
end

function weight_matrix_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat =  get(kwargs, :weightmatrix) do
        error("Connection weight matrix from $name_src to $name_dst is not specified.")
    end
    if size(conn_mat) != (N_src, N_dst)
        error("The connection matrix must be of size $(N_src) x $(N_dst), got $(size(conn_mat))")
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if !iszero(conn_mat[i,j])
                system_wiring_rule!(g, neurons_src[i], neurons_dst[j]; kwargs..., weight=conn_mat[i,j])
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g::PartitioningGraphSystem,
                                           blox::AbstractComposite; kwargs...)
    merge!(g, blox.graph.flat_graph)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractBlox, blox_dst::AbstractBlox; weight, ignore_if_exists=false, kwargs...)
    if ignore_if_exists && has_connection(g, blox_src, blox_dst)
        # if there's already a conntection just bail out early
        return nothing
    end
    # The fallback for all blox: just use BasicConnection
    conn = BasicConnection(weight)
    if blox_src isa AbstractComposite || blox_dst isa AbstractComposite
        name_src = namespaced_nameof(blox_src)
        name_dst = namespaced_nameof(blox_dst)
        error("Tried to connect a composite blox using the fallback wiring rule, but this rule only works for non-composite blox. Source blox: $name_src, Destination blox: $name_dst")
    end
    add_connection!(g, blox_src, conn, blox_dst; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{AbstractNeuron, AbstractNeuralMass},
                                           blox_dst::Union{AbstractNeuron, AbstractNeuralMass};
                                           weight, connection_rule="basic",
                                           ignore_if_exists=false,
                                           kwargs...)
    if ignore_if_exists && has_connection(g, blox_src, blox_dst)
        # if there's already a conntection just bail out early
        return nothing
    end
    conn = if connection_rule == "basic"
        BasicConnection(weight)
    elseif connection_rule == "psp"
        PSPConnection(weight)
    else
        ArgumentError("Unrecognized connection rule type, got $(connection_rule), expected either \"basic\" or \"psp\".")
    end
    add_connection!(g, blox_src, conn, blox_dst; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, ::AbstractActionSelection; kwargs...)
    nothing
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractBlox, ::AbstractActionSelection; kwargs...)
    nothing
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractNeuralMass, blox_dest::NeurobloxBase.AbstractObserver; weight, kwargs...)
    add_connection!(g, blox_src, BasicConnection(weight), blox_dest; weight, kwargs...)
end
