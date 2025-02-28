module GraphDynamicsInterop

using ..Neuroblox:
    Neuroblox,
    get_exci_neurons,
    get_connection_matrix,
    AbstractNeuronBlox,
    NeuralMassBlox,
    HarmonicOscillator,
    JansenRit,
    QIFNeuron,
    IzhikevichNeuron,
    WilsonCowan,
    IFNeuron,
    LIFNeuron,
    HHNeuronExciBlox,
    HHNeuronInhibBlox,
    HHNeuronInhib_MSN_Adam_Blox,
    HHNeuronInhib_FSI_Adam_Blox,
    HHNeuronExci_STN_Adam_Blox,
    HHNeuronInhib_GPe_Adam_Blox,
    WinnerTakeAllBlox,
    namespaced_nameof,
    NGNMM_theta,
    get_namespaced_sys,
    Striatum,
    Striosome,
    Matrisome,
    TAN,
    SNc,
    Noisy,
    NonNoisy,
    KuramotoOscillator,
    CorticalBlox,
    STN,
    Thalamus,
    GPi,
    GPe,
    Striatum_MSN_Adam,
    Striatum_FSI_Adam,
    GPe_Adam,
    STN_Adam,
    PoissonSpikeTrain,
    LIFExciNeuron,
    LIFInhNeuron,
    LIFExciCircuitBlox,
    LIFInhCircuitBlox,
    PINGNeuronExci,
    PINGNeuronInhib,
    AbstractPINGNeuron,
    Connector,
    VanDerPol

using GraphDynamics:
    GraphDynamics,
    GraphSystem,
    ODEGraphSystem,
    SDEGraphSystem,
    Subsystem,
    ConnectionRule,
    ConnectionMatrix,
    ConnectionMatrices,
    NotConnected,
    SubsystemStates,
    SubsystemParams,
    VectorOfSubsystemStates,
    get_tag,
    get_states,
    get_params,
    isstochastic,
    initialize_input,
    combine,
    subsystem_differential,
    StateIndex,
    ParamIndex,
    event_times,
    calculate_inputs

using Random:
    Random,
    default_rng

using StatsBase:
    StatsBase,
    sample

const NB = Neuroblox
const MTK = NB.ModelingToolkit

using SymbolicUtils: Chain, Postwalk

using Graphs
using MetaGraphs

using ModelingToolkit, MetaGraphs, Graphs
const MTK = ModelingToolkit

using ModelingToolkit:
    get_continuous_events,
    get_discrete_events,
    SDESystem,
    ODESystem,
    getdefault

using ModelingToolkit:
    get_ps,
    get_noiseeqs,
    AbstractSystem,
    ODESystem,
    System,
    SDESystem,
    AbstractSystem,
    get_iv,
    parameters

using SparseArrays:
    SparseArrays,
    sparse,
    nnz

using RecursiveArrayTools: ArrayPartition

using Base: @propagate_inbounds
using Base.Iterators: map as imap
using Base.Iterators: filter as ifilter

using Distributions:
    Distributions,
    Bernoulli,
    Poisson

using Accessors:
    Accessors,
    @set,
    @reset

using SciMLBase:
    SciMLBase,
    add_tstop!

using Symbolics:
    Symbolics,
    tosymbol,
    get_variables,
    simplify

include("neuron_interop.jl")
include("connection_interop.jl")

get_blox(g::AbstractMetaGraph, i) = props(g, i)[:blox]
get_subsystem(g::AbstractMetaGraph, i) = props(g, i)[:subsystem]
set_subsystem!(g::AbstractMetaGraph, subsystem, i) = set_prop!(g, i, :subsystem, subsystem)
get_name( g::AbstractMetaGraph, I...) = props(g, I...)[:name]
set_name!(g::AbstractMetaGraph, name, I...) = set_prop!(g, I..., :name, name)

struct GraphSystemBuilder <: AbstractMetaGraph{Int}
    g::MetaDiGraph
    composite_discrete_events_builder::Vector{Any}
    composite_continuous_events_builder::Vector{Any}
end
GraphSystemBuilder() = GraphSystemBuilder(MetaDiGraph(), [], [])

Graphs.edges(g::GraphSystemBuilder) = edges(g.g)
Graphs.vertices(g::GraphSystemBuilder) = vertices(g.g)
Graphs.add_vertex!(g::GraphSystemBuilder) = add_vertex!(g.g)
Graphs.add_vertex!(g::GraphSystemBuilder, k::Symbol, v) = add_vertex!(g.g, k, v)
Graphs.add_vertex!(g::GraphSystemBuilder, d::Dict) = add_vertex!(g.g, d)
Graphs.add_vertices!(g::GraphSystemBuilder, n::Int) = add_vertices!(g.g, n)
Graphs.add_edge!(g::GraphSystemBuilder, i::Integer, j::Integer, d::Dict) = add_edge!(g.g, i, j, d)
Graphs.add_edge!(g::GraphSystemBuilder, i::Integer, j::Integer, k::Symbol, v) = add_edge!(g.g, i, j, k, v)
Graphs.has_edge(g::GraphSystemBuilder, i::Integer, j::Integer) = has_edge(g.g, i, j)
MetaGraphs.props(g::GraphSystemBuilder, i::Integer) = props(g.g, i)
MetaGraphs.props(g::GraphSystemBuilder, i::Integer, j::Integer) = props(g.g, i, j)
MetaGraphs.set_prop!(g::GraphSystemBuilder, i::Integer, s::Symbol, val) = set_prop!(g.g, i, s, val)


function add_subsystem!(g::GraphSystemBuilder, subsystem::T) where {T}
    add_vertex!(g, :subsystem, subsystem)
end
function add_subsystem!(g::GraphSystemBuilder, subsystem::T, name::Symbol) where {T}
    add_vertex!(g, Dict(:subsystem => subsystem, :name => name))
end


"""
    flat_graph(g::MetaDiGraph)

Take some graph describing (potentially heirarchical) connections between
neurons and neuron connections taken from Neuroblox, and then turn it into
a flattened structure where the heirarchy has been flattened down to just
the bottom-level structures, and connections between them.

During this process, we replace the blox and connection rules from Neuroblox
with Subsystems and ConnectionRules from GraphDynamics.
"""
function flat_graph(_g::MetaDiGraph)
    h = GraphSystemBuilder()
    g = copy(_g)
    vertex_map = Dict{Int, Any}()
    h_i = 0

    for g_i ∈ vertices(g)
        h_i = populate_flatgraph(h, g, get_blox(g, g_i), vertex_map, g_i, h_i)
    end
    add_gap_backedges!(g)
    for edg ∈ edges(g)
        i, j = src(edg), dst(edg)
        blox_src, blox_dst = get_blox(g, i), get_blox(g, j)
        blox_wiring_rule!(h, blox_src, blox_dst, vertex_map[i], vertex_map[j], props(g, i, j))
    end
    h
end
function add_gap_backedges!(g, edgs=edges(g))
    for edg ∈ edgs
        # Do this because FSI gap connections are *bidirectional* and thus I need to add the reverse
        # connections, so we add them first before doing up all the wiring rules
        #
        # if other bidirectional connection rules like `:gap` start showing up, this should get split
        # out into its own function
        i, j = src(edg), dst(edg)
        pij = props(g, i, j)
        pji = props(g, j, i)
        if get(pij, :gap, false)
            # check if the mirror connection exists
            if has_edge(g, j, i)
                pji_gap_weight = get(pji, :gap, false) ?  pji[:gap_weight] : 0.0
                set_props!(g, j, i, merge(Dict(:weight => 0.0), # this goes first because we want it to be
                                          #                       overwritten if these keys are in pji
                                          pji,
                                          Dict(:gap => true,
                                               :gap_weight => pji_gap_weight,
                                               :gap_weight_reverse => pij[:gap_weight])))
                # set_prop!(g, j, i, :gap_weight, get(props(g, j, i), :gap_weight, 0.0))
            else
                # if it does not exist, we'll create a whole new connection that has nothing but
                # the reverse connection
                add_edge!(g, j, i, Dict(:weight => 0.0,
                                        :gap => true,
                                        :gap_weight => 0.0,
                                        :gap_weight_reverse => pij[:gap_weight]))
            end
            # now we tell the original connection, and it has a :gap_weight_reverse equal to
            # the mirror connection's :gap_weight (which might be zero)
            set_prop!(g, i, j, :gap_weight_reverse, get(props(g, j, i), :gap_weight, 0.0))
        end
    end
end

function populate_flatgraph(h, g, blox, v, g_i, h_i)
    # This recursive function implementation is completely cursed, but I couldn't figure out a 
    # cleaner way to do it
    # 
    # Idea here is that if I have say a blox composed of blox composed of blox, then I want e.g.
    # v[gi] = [[[h_1, h_2], [h_3, h_4]], [[h_5, h_6], [h_7, h_8, h_9]]]
    # i.e. a vector of vectors of vectors, here h_i is the index of the ith lowest-level blox.
    # these vectors are passed around to blox_wiring_rule! at various points
    oname = outer_nameof(blox)
    if length(components(blox)) == 1 && only(components(blox)) == blox
        h_i += 1
        add_subsystem!(h, to_subsystem(blox), Neuroblox.namespaced_nameof(blox))
        if v isa Dict
            @assert !haskey(v, g_i)
            v[g_i] = h_i
        else
            push!(v, h_i)
        end
    else
        u = []
        if v isa Dict
            @assert !haskey(v, g_i)
            v[g_i] = u
        else
            push!(v, u)
        end
        for comp ∈ components(blox)
            h_i = populate_flatgraph(h, g, comp, u, g_i, h_i)
        end
    end
    vi = if v isa Dict
        v[g_i]
    else
        v[end]
    end
    blox_wiring_rule!(h, blox, vi, props(g, g_i))
    h_i
end

function get_sorted_subsystem_types(subsystems_flat; N_tries=200)
    types = unique(imap(typeof, subsystems_flat))
    for _ ∈ 1:N_tries # Do a really dumb sorting algorithm (since we don't necessarily have a total order)
        modified = false
        for i ∈ eachindex(types)
            type_i = types[i]
            for j ∈ eachindex(types)
                type_j = types[j]
                if GraphDynamics.must_run_before(type_j, type_i) # If j has to run before i, we swap the order of i and j
                    types[i] = type_j
                    types[j] = type_i
                    modified = true
                end
            end
        end
        if !modified # if we passed through all elements without any swaps, then it's ordered and we can return
            return types
        end
    end
    error("""
Could not create a ordered subsystem layout in $(N_tries) attempts, this is likely because you made a non-transitive set of `must_run_before` definitions, e.g.
    must_run_before(::Type{A}, ::Type{B}) where {A, B} = true
    must_run_before(::Type{B}, ::Type{A}) where {B, A} = true
    """)
end

function check_all_supported_blox(g::MetaDiGraph)
    unsupported_blox = filter(vertices(g)) do i
        blox = get_blox(g, i)
        !issupported(blox)
    end
    if !isempty(unsupported_blox)
        v = unique(typeof.(unsupported_blox))
        error("Got unsupported Blox. The GraphDynamics backend is not compatible with blox of type $(join(v, ", "))")
    end
end


"""
    graphsystem_from_graph(g::MetaDiGraph; sparsity_heuristic=1.0, sparse_length_cutoff=0)

Take a graph constructed as found in Neuroblox.jl describing different cortical
components connected together, and then convert it into a form usable with GraphDynamics.jl

For each connection matrix in the connection matrix bundle, `sparsity_heuristic` will determine
how sparse that connection matrix should be before falling back to storing a `SparseMatrixCSC`
of connections, but only if the matrix is also longer than `sparse_length_cutoff` (this is to avoid)
situations where tiny matrices like (e.g. 5x5) get stored as sparse arrays rather than dense arrays. 
"""
function graphsystem_from_graph(_g::MetaDiGraph; sparsity_heuristic=1.0, sparse_length_cutoff=0)
    check_all_supported_blox(_g)
    g = flat_graph(_g)

    subsystems_and_names_flat = map(vertices(g)) do i
        (subsystem = get_subsystem(g, i), name = get_name(g, i))
    end
    names_flat = map(last, subsystems_and_names_flat)
    subsystems_flat = map(first, subsystems_and_names_flat)

    subsystem_types = get_sorted_subsystem_types(subsystems_flat)
    
    index_map = Dict{Int, Tuple{Int, Int}}()
    let js = zeros(Int, length(subsystem_types))
        for (idx, sys) ∈ enumerate(subsystems_flat)
            T = typeof(sys)
            i = findfirst(==(T), subsystem_types)
            js[i] += 1
            index_map[idx] = (i, js[i])
        end
    end
    
    connection_types = (unique ∘ imap)(edges(g)) do e
        typeof(props(g, src(e), dst(e))[:conn])
    end
    NST = length(subsystem_types)
    NCT = length(connection_types)
    tstops = Float64[]

    #todo maybe take advantage of the index_map here
    subsystems_and_names = let i = 1, j = 1
        (Tuple ∘ map)(subsystem_types) do T
            filter(subsystems_and_names_flat) do (;subsystem, name)
                if subsystem isa T
                    for t ∈ event_times(subsystem)
                        @debug "event at" t
                        push!(tstops, t)
                    end
                    true
                else
                    false
                end
            end
        end
    end
    subsystems = map(v -> map(first, v), subsystems_and_names)
    for (idx, sys) ∈ enumerate(subsystems_flat)
        (i, j) = index_map[idx]
        sys == subsystems[i][j] || error("Internal Error: subsystems list and the index_map don't agree, this shouldn't be possible")
    end
    
    connection_matrices = (ConnectionMatrices ∘ Tuple ∘ map)(connection_types) do CT
        (ConnectionMatrix ∘ Tuple ∘ map)(subsystem_types) do T
            vis = [vi for vi ∈ vertices(g) if subsystems_flat[vi] isa T]
            (Tuple ∘ map)(subsystem_types) do U
                vjs = [vj for vj ∈ vertices(g) if subsystems_flat[vj] isa U]
                Is = Int[]
                Js = Int[]
                Vs = CT[]
                for (j, vj) ∈ enumerate(vjs)
                    for (i, vi) ∈ enumerate(vis)
                        if has_edge(g, vi, vj)
                            conn = props(g, vi, vj)[:conn]
                            if conn isa CT
                                push!(Is, i)
                                push!(Js, j)
                                push!(Vs, conn)
                                
                                for t ∈ event_times(conn)
                                    @debug "event at" event_times(conn)
                                    push!(tstops, t)
                                end
                            end
                        end
                    end
                end
                rule_matrix_sparse = sparse(Is, Js, Vs, length(vis), length(vjs))
                sparsity = nnz(rule_matrix_sparse)/length(rule_matrix_sparse)
                if iszero(sparsity)
                    NotConnected()
                elseif sparsity <= sparsity_heuristic && length(rule_matrix_sparse) > sparse_length_cutoff
                    #@info "$CT was sparse" sparsity length = length(rule_matrix_sparse)
                    rule_matrix_sparse
                else
                    #@info "$CT was dense" sparsity length = length(rule_matrix_sparse)
                    collect(rule_matrix_sparse)
                end
            end
        end
    end
    states_partitioned = map(v -> map(get_states, v), subsystems)
    params_partitioned = map(v -> map(get_params, v), subsystems)
    names_partitioned  = map(v -> map(last, v), subsystems_and_names)

    composite_continuous_events_partitioned = let
        if isempty(g.composite_continuous_events_builder)
            nothing
        else
            error("Composite continuous events are not yet implemented")
        end
    end
    composite_discrete_events_partitioned = let
        if isempty(g.composite_discrete_events_builder)
            nothing
        else
            events_flat = map(e -> e(index_map), g.composite_discrete_events_builder)
            distinct_event_types = unique(typeof.(events_flat))
            (Tuple∘map)(distinct_event_types) do ET
                filter(events_flat) do ev
                    if ev isa ET
                        for t ∈ event_times(ev)
                            push!(tstops, t)
                        end
                        true
                    else
                        false
                    end
                end
            end
        end
    end
    gsys_args = (;connection_matrices,
                 states_partitioned,
                 params_partitioned,
                 tstops,
                 composite_discrete_events_partitioned,
                 composite_continuous_events_partitioned,
                 names_partitioned
                 )
    if any(v -> any(isstochastic, v), subsystems)
        SDEGraphSystem(;gsys_args...)
    else
        ODEGraphSystem(;gsys_args...)
    end
end

end#module GraphDynamicsInterop

