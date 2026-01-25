module NeurobloxPharma

using Reexport

@reexport using NeurobloxBase

using NeurobloxBase:
    AbstractNeuron,
    AbstractExciNeuron,
    AbstractInhNeuron, 
    AbstractComposite,
    AbstractNeuralMass,
    AbstractReceptor,
    AbstractModulator,
    AbstractDiscrete,
    AbstractStimulus, 
    AbstractLearningRule,
    NoLearningRule,
    AbstractAgent, 
    AbstractEnvironment,
    AbstractActionSelection,
    AbstractExperimentMonitor,
    update_monitor!,
    AbstractModulator,
    AbstractDiscrete,
    AbstractStimulus,
    AbstractSimpleStimulus,
    AbstractLearningRule,
    NoLearningRule,
    AbstractAgent,
    AbstractEnvironment,
    AbstractActionSelection,
    BasicConnection,
    EventConnection,
    t_block_event,
    maybe_set_state_pre,
    maybe_set_state_post,
    apply_learning_rules!

using SciMLBase: SciMLBase, AbstractSolution, remake

using DataFrames: DataFrames, DataFrame, Not

using CSV: CSV, read, write

using LogExpFunctions: logistic

using Distributions:
    Distributions,
    Uniform,
    Poisson,
    Bernoulli
using Base: isstored
using Random
using Random: default_rng

using ProgressMeter:
    ProgressMeter,
    Progress,
    next!,
    finish!

using NeurobloxBase:
    connect_action_selection!,
    run_warmup,
    run_trial!,
    run_experiment!,
    get_trial_stimulus,
    weight_gradient

using GraphDynamics:
    GraphDynamics,
    NotConnected,
    ConnectionRule,
    ConnectionMatrices,
    ConnectionMatrix,
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
    get_parent_index

using StatsBase: StatsBase, sample

using Accessors:
    Accessors,
    @set,
    @reset

using SciMLBase: SciMLBase, solve

using DiffEqCallbacks: DiffEqCallbacks, PeriodicCallback

using NaNMath




include("neurons.jl")
include("neural_mass.jl")
include("discrete.jl")
include("cortical.jl")
include("subcortical.jl")
include("sources.jl")
include("receptors.jl")
include("receptors_support.jl")
include("connections.jl")
include("reinforcement_learning.jl")

export HHNeuronExci, HHNeuronInhib, HHNeuronFSI
export GABA_A_Synapse, Glu_AMPA_Synapse, Glu_AMPA_STA_Synapse
export NGNMM_theta, NextGenerationEI
export ImageStimulus, PulsesInput, VoltageClampSource
export WinnerTakeAll, Cortical
export Matrisome, Striosome, Striatum, GPi, GPe, Thalamus, STN, TAN, SNc
export HebbianPlasticity, HebbianModulationPlasticity
export Agent, ClassificationEnvironment, GreedyPolicy
export run_warmup, run_trial!, run_experiment!
export ProgressMeterMonitor, get_ff_inh_neurons
export synapse_wiring_rule!

export
    # Neurons
    BaxterSensoryNeuron, TRNNeuron, MuscarinicNeuron,
    VTADANeuron, VTAGABANeuron,
    # Receptors
    MoradiNMDAR, MoradiFullNMDAR,
    GABA_B_Synapse, NMDA_Synapse,
    MsnNMDAR, MsnAMPAR,
    MsnD1Receptor, MsnD2Receptor,
    HTR5, MuscarinicR,
    Alpha7ERnAChR, CaTRPM4R, Beta2nAChR,
    # Support inputs
    ConstantDAInput, ConstantIStimInput, ConstantIAppInput,
    ConstantAChInput, ConstantNicInput, ConstantICaInput,
    ConstantCaBulkInput, ConstantCChInput, ConstantModeInput,
    ConstantMuscarinicInput, ConstantVPreInput, ConstantVPostInput,
    ConstantMNMDA1Input, ConstantGAsympInput, ConstantMAMPA2Input

end
