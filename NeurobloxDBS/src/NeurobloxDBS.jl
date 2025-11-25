module NeurobloxDBS

using Reexport

@reexport using NeurobloxBase

using NeurobloxBase: AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, 
                    AbstractComposite, AbstractNeuralMass, AbstractReceptor,
                    AbstractObserver, AbstractModulator, AbstractStimulus, AbstractSimpleStimulus,
                    AbstractLearningRule, NoLearningRule, AbstractAgent, AbstractEnvironment, AbstractActionSelection

import NeurobloxBase: Connector

using ModelingToolkit: getp

using SciMLBase: SciMLBase, AbstractSolution

using StatsBase: sample

include("neurons.jl")
include("composites.jl")
include("sources.jl")
include("connections.jl")
include("GraphDynamicsInterop/GraphDynamicsInterop.jl")

export HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam,
    HHNeuronInhib_GPe_Adam, Striatum_MSN_Adam, Striatum_FSI_Adam, GPe_Adam, STN_Adam
export DBS, ProtocolDBS, SquareStimulus, BurstStimulus 
export detect_transitions, compute_transition_times, compute_transition_values, get_protocol_duration, get_stimulus_function
export square
end
