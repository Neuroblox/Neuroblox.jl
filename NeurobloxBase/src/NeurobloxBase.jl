module NeurobloxBase
import Base: merge

using Base.Threads: nthreads

using OhMyThreads: tmapreduce

using Reexport
@reexport using ModelingToolkit
@reexport using ModelingToolkit: ModelingToolkit.t_nounits as t, ModelingToolkit.D_nounits as D

@reexport import Graphs: add_edge!
@reexport using MetaGraphs: MetaDiGraph, get_prop, props

using Graphs

using LinearAlgebra

using DSP, Statistics

using Interpolations
using Random
using Random: default_rng
using OrderedCollections

using StatsBase: sample
using Distributions

using SciMLBase: SciMLBase, AbstractSolution, solve, remake

using ModelingToolkit: get_namespace, get_systems, isparameter,
                    renamespace, namespace_equation, namespace_parameters, namespace_expr,
                    AbstractODESystem, VariableTunable, getp
import ModelingToolkit: equations, inputs, outputs, unknowns, parameters, discrete_events, nameof, getdescription

using Symbolics: @register_symbolic, getdefaultval, get_variables

using GraphDynamics: GraphDynamics, GraphSystem, add_connection!, add_node!, PartitionedGraphSystem, nodes

using Peaks: argmaxima, peakproms!, peakheights!, findmaxima

using SparseArrays

using Preferences: Preferences

abstract type AbstractBlox end
abstract type AbstractNeuralMass <: AbstractBlox end
abstract type AbstractComposite <: AbstractBlox end
abstract type AbstractStimulus <: AbstractBlox end
"""
AbstractSimpleStimulus are continuous Blox representing stimuli that only have one output variable.
"""
abstract type AbstractSimpleStimulus <: AbstractStimulus end
abstract type AbstractDiscrete <: AbstractBlox end
abstract type AbstractModulator <: AbstractDiscrete end
abstract type AbstractObserver end # not AbstractBlox since it should not show up in the GUI

abstract type AbstractComponent end # TO REMOVE
abstract type BloxConnection end # TO REMOVE
abstract type BloxUtilities end # TO REMOVE
abstract type Merger end # TO REMOVE

abstract type AbstractNeuron <: AbstractBlox end
abstract type AbstractInhNeuron <: AbstractNeuron end
abstract type AbstractExciNeuron <: AbstractNeuron end
abstract type AbstractReceptor <: AbstractBlox end

abstract type SpectralUtilities <: BloxUtilities end # TO REMOVE

abstract type BloxConnectFloat <: BloxConnection end # TO REMOVE
abstract type BloxConnectComplex <: BloxConnection end # TO REMOVE
abstract type BloxConnectMultiFloat <: BloxConnection end # TO REMOVE
abstract type BloxConnectMultiComplex <: BloxConnection end # TO REMOVE

abstract type AbstractAgent end
abstract type AbstractEnvironment end
abstract type AbstractLearningRule end
abstract type AbstractActionSelection <: AbstractBlox end
struct NoLearningRule <: AbstractLearningRule end

abstract type AbstractExperimentMonitor end

Para_dict = Dict{Symbol, Union{<: Real, Num}}

include("connector.jl")
include("utils.jl")
include("reinforcement_learning.jl")
include("system_from_graph.jl")
include("adjacency.jl")
include("GraphDynamicsInterop/GraphDynamicsInterop.jl")
include("graph_macro.jl")
using .GraphDynamicsInterop: to_metadigraph

const Neuron = AbstractNeuron

"""
random_initials creates a vector of random initial conditions for an ODESystem that is
composed of a list of blox.  The function finds the initial conditions in the blox and then
sets a random value in between range tuple given for that state.

It has the following inputs:
    odesys: ODESystem
    blox  : list of blox

And outputs:
    u0 : Float64 vector of initial conditions
"""
function random_initials(odesys::ODESystem, blox) # TO REMOVE
    odestates = unknowns(odesys)
    u0 = Float64[]
    init_dict = Dict{Num,Tuple{Float64,Float64}}()

    # first merge all the inital dicts into one
    for b in blox
        merge!(init_dict, b.initial)
    end

    for state in odestates
        init_tuple = init_dict[state]
        push!(u0, rand(Distributions.Uniform(init_tuple[1],init_tuple[2])))
    end
    
    return u0
end

function print_license()
    printstyled("Important Note: ", bold = true)
    print("""Neuroblox is a commercial product of Neuroblox, Inc.
It is free to use for non-commercial academic teaching
and research purposes. For commercial users, license fees apply.
Please refer to the End User License Agreement
(https://github.com/Neuroblox/NeurobloxEULA) for details.
Please contact sales@neuroblox.org for purchasing information.

To report any bugs, issues, or feature requests for Neuroblox software,
please use the public Github repository NeurobloxIssues, located at
https://github.com/Neuroblox/NeurobloxIssues.
""")
end

"""
    meanfield(blox, sol)

Plot the mean-field voltage (in mV) as a function of time (in ms) for a blox. 

Note: this function requires Makie to be loaded.

See also [`meanfield!`](@ref), [`meanfield_timeseries`](@ref).
"""
function meanfield end

"""
    meanfield!(ax::Axis, blox, sol)

Update an existing plot to show the mean-field voltage (in mV) as a function of time (in ms) for a blox. 

Note: this function requires Makie to be loaded.

See also [`meanfield!`](@ref), [`meanfield_timeseries`](@ref).
"""
function meanfield! end

"""
    rasterplot(blox, sol; threshold = nothing, kwargs...)

Create a scatterplot of neuron firing events, where the x-axis is time (in ms) and the y-axis is the neuron's index. Internally calls [`detect_spikes`](@ref), and the `threshold` kwarg is propagated to `detect_spikes`, while the rest of the kwargs are Makie kwargs.

Note: this function requires Makie to be loaded.

See also [`rasterplot!`](@ref), [`detect_spikes`](@ref).
"""
function rasterplot end

"""
    rasterplot!(ax::Axis, blox, sol; threshold = nothing, kwargs...)

Update an existing plot to show a scatterplot of neuron firing events, where the x-axis is time (in ms) and the y-axis is the neuron's index. Internally calls [`detect_spikes`](@ref). The `threshold` kwarg is the voltage threshold for `detect_spikes`, while the rest of the kwargs are Makie kwargs.

Note: this function requires Makie to be loaded.

See also [`rasterplot!`](@ref), [`detect_spikes`](@ref).
"""
function rasterplot! end

"""
    stackplot(blox, sol)

Plot the voltage timeseries of the neurons in a Blox, stacked on top of each other.

Note: this function requires Makie to be loaded.

See also [`stackplot!`](@ref).
"""
function stackplot end

"""
    stackplot!(ax::Axis, blox, sol)

Update an existing plot to show the voltage timeseries of the neurons in a Blox, stacked on top of each other.

Note: this function requires Makie to be loaded.

See also [`stackplot`](@ref).
"""
function stackplot! end

"""
    frplot(blox, sol; win_size = 10, overlap = 0, transient = 0, threshold = nothing, kwargs...)

Plot the firing frequency (either individual firing frequency for a neuron or mean firing frequency for a population) of a blox as a function of time (in s). The named keyword arguments are propagated to [`firing_rate`](@ref), while the rest of the kwargs are propagated to Makie for plotting.

Note: this function requires Makie to be loaded.

See also [`frplot!`](@ref), [`firing_rate`](@ref).
"""
function frplot end

"""
    frplot!(ax::Axis, blox, sol; win_size = 10, overlap = 0, transient = 0, threshold = nothing, kwargs...)

Update an existing plot with the firing frequency (either individual firing frequency for a neuron or mean firing frequency for a population) of a blox as a function of time (in s). The named keyword arguments are propagated to [`firing_rate`](@ref), while the rest of the kwargs are propagated to Makie for plotting.

Note: this function requires Makie to be loaded.

See also [`frplot`](@ref), [`firing_rate`](@ref).
"""
function frplot! end

"""
    voltage_stack(blox, sol; kwargs...)

Create and display a [`stackplot`](@ref) of the voltage timeseries.

Note: this function requires Makie to be loaded.
"""
function voltage_stack end

"""
    powerspectrumplot(blox, sol; sampling_rate = nothing, method = nothing, window = nothing, kwargs...)

Plot the power spectrum of the solution (intensity as a function of frequency). The named keyword arguments are propagated to the internal [`powerspectrum`](@ref) call, while the rest of the keyword arguments are propagated to Makie for plotting.

Note: this function requires Makie to be loaded.

See also [`powerspectrumplot!`](@ref), [`powerspectrum`](@ref).
"""
function powerspectrumplot end

"""
    powerspectrumplot!(ax::Axis, blox, sol; sampling_rate = nothing, method = nothing, window = nothing, kwargs...)

Update an existing plot with the power spectrum of the solution (intensity as a function of frequency). The named keyword arguments are propagated to the internal [`powerspectrum`](@ref) call, while the rest of the keyword arguments are propagated to Makie for plotting.

Note: this function requires Makie to be loaded.

See also [`powerspectrumplot`](@ref), [`powerspectrum`](@ref).
"""
function powerspectrumplot! end

function adjacency end
function adjacency! end

function __init__()
    if parse(Bool, Preferences.@load_preference("PrintLicense", "true"))
        print_license()
    end
end

export Neuron, AbstractBlox, AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, AbstractComposite, AbstractReceptor, AbstractNeuralMass, AbstractModulator, AbstractDiscrete, AbstractStimulus
export AbstractAgent, AbstractEnvironment, AbstractLearningRule, NoLearningRule, AbstractActionSelection, AbstractExperimentMonitor
export Neuron, AbstractBlox, AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, AbstractComposite, AbstractReceptor, AbstractNeuralMass, AbstractModulator, AbstractDiscrete, AbstractStimulus, AbstractSimpleStimulus
export AbstractAgent, AbstractEnvironment, AbstractLearningRule, NoLearningRule, AbstractActionSelection
export AdjacencyMatrix, Connector, generate_weight_param, generate_gap_weight_param, indegree_constrained_connections
export connection_rule, connection_equations, connection_spike_affects, connection_callbacks
export equations, discrete_callbacks, sources, destinations, weights, delays, spike_affects, learning_rules
export add_blox!, get_system
export system_from_graph, system_from_parts, connectors_from_graph, action_selection_from_graph, system, graph_delays, generate_discrete_callbacks
export create_adjacency_edges!, adjmatrixfromdigraph
export get_weights, get_namespaced_sys, get_neurons, get_exci_neurons, get_inh_neurons, get_neuron_color
export detect_spikes, firing_rate, inter_spike_intervals, flat_inter_spike_intervals, powerspectrum
export voltage_timeseries, meanfield_timeseries, state_timeseries
export nameof, namespaced_name, namespaced_nameof, narrowtype, changetune
export maybe_set_state_pre!, maybe_set_state_post!, connect_action_selection!
export weight_gradient, learning_rules, learning_rules_from_graph, get_trial_stimulus
export run_warmup, run_experiment!, run_trial!, reset!, increment_trial!, update_monitor!
export meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, frplot, frplot!, voltage_stack, adjacency, adjacency!
export powerspectrumplot, powerspectrumplot!, welch_pgram, periodogram, hanning, hamming
export inputs, outputs, equations, unknowns, parameters, discrete_events
export get_parts, get_neurons, get_exci_neurons, get_inh_neurons, get_namespaced_sys, get_weight
export get_system, get_connection_matrix, get_components, get_dynamics_components, get_weightmatrix
export get_neuron_color, get_discrete_parts, get_gap, get_gap_weight, get_learning_rule
export get_density, get_connection_rule, get_states_spikes_affect, get_params_spikes_affect, get_event_time
export paramscoping, density_connections, hypergeometric_connections, weight_matrix_connections
export make_unique_param_pairs, to_vector, params_with_substrings, states_with_substrings
export get_blox_by_name
export make_unique_param_pairs, to_vector
export param_symbols, state_symbols, input_symbols, output_symbols, computed_state_symbols
export @graph
end
