module NeurobloxDBS

using Reexport

@reexport using NeurobloxBase

using NeurobloxBase: AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, 
                    AbstractComposite, AbstractNeuralMass, AbstractReceptor,
                    AbstractObserver, AbstractModulator, AbstractStimulus, AbstractSimpleStimulus,
                    AbstractLearningRule, NoLearningRule, AbstractAgent, AbstractEnvironment, AbstractActionSelection,
                    BasicConnection, HHConnection_GAP, HHConnection_GAP_Reverse, EventConnection, ReverseConnection, 
                    GraphDynamics, inner_namespaceof, namespaced_name

using GraphDynamics: 
    GraphDynamics,
    GraphSystem,
    GraphSystemConnection,
    Subsystem,
    ConnectionRule,
    ConnectionMatrix,
    ConnectionMatrices,
    NotConnected,
    SubsystemStates,
    SubsystemParams,
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
    partitioned,
    maybe_sparse_enumerate_col

using SciMLBase: SciMLBase, AbstractSolution

using StatsBase: sample

include("neurons.jl")
include("composites.jl")
include("sources.jl")
include("connections.jl")

export HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam,
    HHNeuronInhib_GPe_Adam, Striatum_MSN_Adam, Striatum_FSI_Adam, GPe_Adam, STN_Adam
export DBS, ProtocolDBS, SquareStimulus, BurstStimulus 
export DBSConnection
export detect_transitions, compute_transition_times, compute_transition_values, get_protocol_duration, get_stimulus_function
export square
end
