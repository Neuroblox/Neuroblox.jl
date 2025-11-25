module GraphDynamicsInterop

using NeurobloxBase.GraphDynamicsInterop:
    define_neuron,
    @recursive_getdefaults, 
    BasicConnection,
    HHConnection_GAP,
    HHConnection_GAP_Reverse,
    EventConnection,
    ReverseConnection, 
    t_block_event,
    maybe_set_state_pre,
    maybe_set_state_post,
    apply_learning_rules!

using NeurobloxPharma.GraphDynamicsInterop:
    hypergeometric_connections!,
    weight_matrix_connections!,
    find_competitor_matrisome

using NeurobloxPharma: increment_pixel!

using ..NeurobloxSynapticBlox:
    HHExci,
    HHInhi,
    GABA_A_Synapse,
    Glu_AMPA_Synapse,
    Glu_AMPA_STA_Synapse,
    NGNMM_theta,
    Matrisome,
    Striosome,
    TAN,
    SNc,
    WinnerTakeAll,
    Cortical,
    Striatum,
    STN,
    GPi,
    GPe,
    Thalamus,
    ImageStimulus,
    HebbianPlasticity,
    HebbianModulationPlasticity,
    @named,
    get_synapse!

using NeurobloxBase:
    NeurobloxBase,
    namespaced_name,
    namespaced_nameof,
    get_namespaced_sys,
    get_event_time,
    NeurobloxBase,
    AbstractNeuron,
    AbstractExciNeuron,
    AbstractInhNeuron,
    AbstractReceptor,
    AbstractComposite,
    Connector,
    NoLearningRule,
    get_weight,
    get_density,
    to_metadigraph,
    connectors_from_graph,
    system_from_graph,
    system_from_parts,
    get_connection_matrix,
    connect_action_selection!,
    AbstractActionSelection


using ModelingToolkit:
    @parameters,
    @variables,
    System,
    ODESystem,
    Num,
    @named

using Accessors:
    Accessors,
    @set,
    @reset

using Distributions:
    Distributions,
    Bernoulli,
    Uniform,
    Poisson


using GraphDynamics:
    GraphDynamics,
    NotConnected,
    ConnectionRule,
    ConnectionMatrices,
    ConnectionMatrix,
    PartitionedGraphSystem,
    GraphSystem,
    initialize_input,
    subsystem_differential,
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    isstochastic,
    event_times,
    get_tag,
    get_states,
    get_params,
    calculate_inputs,
    partitioned,
    maybe_sparse_enumerate_col,
    calculate_inputs,
    to_subsystem,
    nodes,
    add_connection!,
    connections,
    has_connection,
    system_wiring_rule!,
    make_connection_matrices

@eval begin
    for sys ∈ [HHExci(name=:hhe)
               HHInhi(name=:hhi)
               GABA_A_Synapse(name=:gas)
               Glu_AMPA_Synapse(name=:gams)
               Glu_AMPA_STA_Synapse(name=:gamss)
               ]
        define_neuron(sys; mod=@__MODULE__())
    end
end

NeurobloxBase.GraphDynamicsInterop.has_t_block_event(::Type{HHExci}) = true
NeurobloxBase.GraphDynamicsInterop.is_t_block_event_time(::Type{HHExci}, key, t) = key == :t_block_late
NeurobloxBase.GraphDynamicsInterop.t_block_event_requires_inputs(::Type{HHExci}) = false
function NeurobloxBase.GraphDynamicsInterop.apply_t_block_event!(vstates, _, s::Subsystem{HHExci}, _, _)
    vstates[:spikes_window] = 0.0
end


include("connections.jl")


get_exci_neurons(blox_src::AbstractComposite) = [blox for blox ∈ nodes(blox_src.graph) if blox isa HHExci]
get_inhi_neurons(blox_src::AbstractComposite) = [blox for blox ∈ nodes(blox_src.graph) if blox isa HHInhi]

end
