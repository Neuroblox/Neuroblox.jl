module GraphDynamicsInterop

using ..Neuroblox:
    Neuroblox,
    D,
    t,
    get_exci_neurons,
    get_inh_neurons,
    get_matrisome,
    get_striosome,
    get_connection_matrix,
    get_inh_neurons,
    get_ff_inh_neurons,
    get_wtas,
    get_ff_inh_num,
    get_connection_rule,
    AbstractBlox,
    AbstractNeuronBlox,
    AbstractDiscrete,
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
    namespaced_name,
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
    LateralAmygdalaCluster,
    LateralAmygdala,
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
    VanDerPol,
    MoradiNMDAR,
    AbstractReceptor,
    ImageStimulus,
    increment_pixel!,
    StimulusBlox,
    CompositeBlox,
    PulsesInput,
    to_vector
    
using ..Neuroblox:
    AbstractEnvironment,
    AbstractLearningRule,
    NoLearningRule,
    HebbianPlasticity,
    HebbianModulationPlasticity,
    weight_gradient,
    get_eval_times,
    get_eval_states,
    dlogistic,
    ClassificationEnvironment,
    increment_trial!,
    reset!,
    get_trial_stimulus,
    AbstractActionSelection,
    GreedyPolicy,
    connect_action_selection!,
    get_eval_states,
    get_eval_times,
    Agent,
    run_experiment!,
    run_warmup,
    run_trial!,
    add_blox!,
    add_edge!,
    get_learning_rule,
    action_selection_from_graph


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
    remake

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

using ProgressMeter:
    ProgressMeter,
    Progress,
    next!,
    finish!


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
include("sources_interop.jl")
include("connection_interop.jl")
include("learning_interop.jl")

end#module GraphDynamicsInterop

