module GraphDynamicsInterop

using ..NeurobloxBase:
    D,
    t,
    get_exci_neurons,
    get_inh_neurons,
    get_connection_matrix,
    get_inh_neurons,
    get_connection_rule,
    get_parts,
    namespaced_nameof,
    namespaced_name,
    get_namespaced_sys,
    to_vector,
    increment_trial!,
    reset!,
    get_trial_stimulus,
    run_experiment!,
    run_warmup,
    run_trial!,
    add_blox!,
    add_edge!,
    get_learning_rule,
    action_selection_from_graph,
    connect_action_selection!,
    system_from_graph,
    weight_gradient

using ..NeurobloxBase:
    NeurobloxBase,
    AbstractBlox,
    AbstractNeuron,
    AbstractDiscrete,
    AbstractNeuralMass,
    AbstractStimulus,
    AbstractComposite,
    AbstractReceptor,
    AbstractEnvironment,
    AbstractLearningRule,
    NoLearningRule,
    AbstractActionSelection,
    AbstractAgent,
    Connector

using GraphDynamics:
    GraphDynamics,
    GraphSystem,
    PartitionedGraphSystem,
    GraphSystemConnection,
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
    event_times,
    calculate_inputs,
    to_subsystem,
    nodes,
    add_connection!,
    add_node!,
    connections,
    has_connection,
    system_wiring_rule!,
    make_connection_matrices,
    partitioned,
    maybe_sparse_enumerate_col

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using Random:
    Random,
    default_rng

using StatsBase:
    StatsBase,
    sample

using SymbolicUtils: Chain, Postwalk

using SymbolicIndexingInterface: variable_symbols 

using Graphs
using MetaGraphs

using ModelingToolkit
const MTK = ModelingToolkit

using ModelingToolkit:
    get_continuous_events,
    get_discrete_events,
    getu,
    SDESystem,
    ODESystem,
    getdefault,
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

using Base: @propagate_inbounds, isstored
using Base.Iterators: map as imap
using Base.Iterators: filter as ifilter

using Distributions:
    Distributions,
    Bernoulli,
    Poisson,
    Uniform

using Accessors:
    Accessors,
    @set,
    @reset

using SciMLBase:
    SciMLBase,
    add_tstop!,
    remake,
    successful_retcode

using Symbolics:
    Symbolics,
    tosymbol,
    get_variables,
    simplify

using StatsBase:
    StatsBase,
    countmap

using DiffEqCallbacks:
    DiffEqCallbacks,
    PeriodicCallback

function get_weight(kwargs, name_src, name_dst)
    get(kwargs, :weight) do
        error("No connection weight specified between $name_src and $name_dst")
    end
end

function get_density(kwargs, name_src, name_dst)
    density = get(kwargs, :density) do
        error("No connection density specified between $name_src and $name_dst")
    end
end

function to_metadigraph(g::GraphSystem)
    g2 = MetaDiGraph()
    d = Dict{Any, Int}()
    i = 0
    for node ∈ nodes(g)
        add_blox!(g2, node)
        d[node] = (i += 1)
    end
    for (; src, dst, data) ∈ connections(g)
        add_edge!(g2, d[src], d[dst], Dict(pairs(data)...))
    end
    g2
end

include("neuron_interop.jl")
include("discrete_interop.jl")
include("connection_interop.jl")
include("learning_interop.jl")

export define_neuron, @recursive_getdefaults, maybe_set_state_pre, maybe_set_state_post

end
