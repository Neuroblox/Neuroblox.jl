module NeurobloxBase
import Base: merge

using Base.Threads: nthreads

using OhMyThreads: tmapreduce

using Reexport
@reexport using GraphDynamics
using SymbolicIndexingInterface:
    SymbolicIndexingInterface,
    variable_symbols,
    parameter_symbols,
    setsym,
    setp,
    setu,
    setsym,
    setp,
    setu,
    getu,
    getp,
    setp,
    getsym
    
using JLD2: JLD2, save, load

using LinearAlgebra

using DSP, Statistics

using Interpolations
using Random
using Random: default_rng
using OrderedCollections

using StatsBase: sample
using Distributions

using SciMLBase: SciMLBase, AbstractSolution, solve, remake

import Base: nameof

using GraphDynamics: GraphDynamics, GraphSystem, add_connection!, add_node!, ConnectionRule, Subsystem, initialize_input, nodes

using Peaks: argmaxima, peakproms!, peakheights!, findmaxima

using SparseArrays

using Preferences: Preferences

using Base: isexpr

using GraphDynamics:
    GraphDynamics,
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    computed_properties,
    computed_properties_with_inputs,
    system_wiring_rule!,
    add_node!,
    add_connection!,
    partitioned,
    calculate_inputs,
    maybe_sparse_enumerate_col,
    nodes

using Accessors: Accessors, @reset, @set
    
using ExproniconLite: ExproniconLite, Substitute
using MLStyle: MLStyle, @match

using DiffEqCallbacks:
    DiffEqCallbacks,
    PeriodicCallback

abstract type AbstractBlox end
abstract type AbstractNeuralMass <: AbstractBlox end
abstract type AbstractComposite <: AbstractBlox end
abstract type AbstractStimulus <: AbstractBlox end
abstract type AbstractLearningRule end
struct NoLearningRule <: AbstractLearningRule end

"""
AbstractSimpleStimulus are continuous Blox representing stimuli that only have one output variable.
"""
abstract type AbstractSimpleStimulus <: AbstractStimulus end
abstract type AbstractDiscrete <: AbstractBlox end
abstract type AbstractModulator <: AbstractDiscrete end
abstract type AbstractObserver end # not AbstractBlox since it should not show up in the GUI

abstract type AbstractNeuron <: AbstractBlox end
abstract type AbstractInhNeuron <: AbstractNeuron end
abstract type AbstractExciNeuron <: AbstractNeuron end
abstract type AbstractReceptor <: AbstractBlox end

abstract type AbstractAgent end
abstract type AbstractEnvironment end

abstract type AbstractActionSelection <: AbstractBlox end
abstract type AbstractExperimentMonitor end

include("utils.jl")
include("connections.jl")
include("reinforcement_learning.jl")
include("adjacency.jl")
include("blox_macro.jl")
include("graph_macro.jl")
include("connection_macro.jl")
include("discrete.jl")
include("save_graph.jl")
include("experiment.jl")

const Neuron = AbstractNeuron

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
    meanfield(blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution)

Plot the mean-field voltage (in mV) as a function of time (in ms) for a blox.

Arguments:
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Note: this function requires Makie to be loaded.

See also [`meanfield!`](@ref), [`meanfield_timeseries`](@ref).
"""
function meanfield end

"""
    meanfield!(ax::Axis, blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol)

Update an existing plot `Axis` to show the mean-field voltage (in mV) as a function of time (in ms) for a blox. 

Arguments:
- ax : an `Axis` object attached to a `Figure` from `Makie`.
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Note: this function requires Makie to be loaded.

See also [`meanfield`](@ref), [`meanfield_timeseries`](@ref).
"""
function meanfield! end

"""
    rasterplot(blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution; threshold = nothing, kwargs...)

Create a scatterplot of spikes, where the x-axis is time (in ms) and the y-axis represents neurons organized in separate rows. 

Arguments:
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- threshold : [mV] Spiking threshold. Internally used in [`detect_spikes`](@ref) to calculate all spiking events. 
            Note that neurons like [`NeurobloxPharma.HHNeuronExci`](@ref) and [`NeurobloxPharma.HHNeuronInhib`](@ref) do not inherently contain a threshold and so require this `threshold` argument to be passed in order to determine spiking events.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

Note: this function requires Makie to be loaded.

See also [`rasterplot!`](@ref), [`detect_spikes`](@ref).
"""
function rasterplot end

"""
    rasterplot!(ax::Axis, blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution; threshold = nothing, kwargs...)

Update an existing plot `Axis` to show a scatterplot of spikes, where the x-axis is time (in ms) and the y-axis represents neurons organized in separate rows. 

Arguments:
- ax : an `Axis` object attached to a `Figure` from `Makie`.
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- threshold : [mV] Spiking threshold. Internally used in [`detect_spikes`](@ref) to calculate all spiking events. 
            Note that neurons like [`NeurobloxPharma.HHNeuronExci`](@ref) and [`NeurobloxPharma.HHNeuronInhib`](@ref) do not inherently contain a threshold and so require this `threshold` argument to be passed in order to determine spiking events.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

Note: this function requires Makie to be loaded.

See also [`rasterplot`](@ref), [`detect_spikes`](@ref).
"""
function rasterplot! end

"""
    stackplot(blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution)

Plot the voltage timeseries of all neurons contained in `blox`, stacked on top of each other.
The x-axis is time (in ms) and on the y-axis voltage timeseries are plotted on separate rows for each neuron in `blox`. 

Arguments:
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Note: this function requires Makie to be loaded.

See also [`stackplot!`](@ref).
"""
function stackplot end

"""
    stackplot!(ax::Axis, blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution)

Update an existing plot `Axis` to show the voltage timeseries of all neurons contained in `blox`, stacked on top of each other.
The x-axis is time (in ms) and on the y-axis voltage timeseries are plotted on separate rows for each neuron in `blox`. 

Arguments:
- ax : an `Axis` object attached to a `Figure` from `Makie`.
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Note: this function requires Makie to be loaded.

See also [`stackplot`](@ref).
"""
function stackplot! end

"""
    frplot(blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution; win_size = 10, overlap = 0, transient = 0, threshold = nothing, kwargs...)

Plot the firing rate of `blox` as a function of time (in s).

Arguments:
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
        If blox contains more than one neurons, then the average firing rate across all neurons is plotted.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- win_size : [ms] Sliding window size.
- overlap : in range [0,1]. Overlap between two consecutive sliding windows.
- transient : [ms] Transient period in the beginning of the timeseries that is ignored during firing rate calculation.
- threshold : [mV] Spiking threshold.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

`win_size`, `overlap`, `transient` and `threshold` are internally used in [`firing_rate`](@ref) to calculate the firing rate timeseries.

Note: this function requires Makie to be loaded.

See also [`frplot!`](@ref), [`firing_rate`](@ref).
"""
function frplot end

"""
    frplot!(ax::Axis, blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution; win_size = 10, overlap = 0, transient = 0, threshold = nothing, kwargs...)

Update an existing plot `Axis` to show the firing rate of `blox` as a function of time (in s).

Arguments:
- ax : an `Axis` object attached to a `Figure` from `Makie`.
- blox : a composite blox, e.g. a brain region or microcircuit, containing multiple neurons or a vector of neurons.
        If blox contains more than one neurons, then the average firing rate across all neurons is plotted.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- win_size : [ms] Sliding window size.
- overlap : in range [0,1]. Overlap between two consecutive sliding windows.
- transient : [ms] Transient period in the beginning of the timeseries that is ignored during firing rate calculation.
- threshold : [mV] Spiking threshold.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

`win_size`, `overlap`, `transient` and `threshold` are internally used in [`firing_rate`](@ref) to calculate the firing rate timeseries.

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
    powerspectrumplot(blox::Union{AbstractComposite, Vector{Union{AbstractNeuron, AbstractNeuralMass}}, sol::AbstractSolution; sampling_rate=nothing, method=periodogram, window=nothing, kwargs...)

Plot the power spectrum (intensity as a function of frequency) of neurons and neural mass objects contained in `blox`. 
    
Arguments:
- blox : a composite blox, e.g. a brain region or microcircuit, or a vector containing multiple neurons and/or neural masses.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- sampling_rate : [Hz] Sampling rate. 
- method : Method to calculate the periodogram. Options are [`periodogram`, `welch_pgram`]. See the [`DSP documentation`](https://docs.juliadsp.org/stable/periodograms/#Periodograms-periodogram-estimation) for more information.
- window : An optional window function to be applied to the original signal before computing the Fourier transform. Options are [`nothing`, `hamming`, `hanning`]. See the [`DSP documentation`](https://docs.juliadsp.org/stable/periodograms/#Periodograms-periodogram-estimation) for more information.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

Note: this function requires Makie to be loaded.

See also [`powerspectrumplot!`](@ref), [`powerspectrum`](@ref).
"""
function powerspectrumplot end

"""
    powerspectrumplot!(ax::Axis, blox::Union{AbstractComposite, Vector{Union{AbstractNeuron, AbstractNeuralMass}}, sol::AbstractSolution; sampling_rate=nothing, method=periodogram, window=nothing, kwargs...)

Update an existing plot `Axis` to show the power spectrum (intensity as a function of frequency) of neurons and neural mass objects contained in `blox`. 
    
Arguments:
- ax : an `Axis` object attached to a `Figure` from `Makie`.
- blox : a composite blox, e.g. a brain region or microcircuit, or a vector containing multiple neurons and/or neural masses.
- sol : a solution object after running a simulation (i.e. the output of `solve`) 

Keyword arguments:
- sampling_rate : [Hz] Sampling rate. 
- method : Method to calculate the periodogram. Options are [`periodogram`, `welch_pgram`]. See the [`DSP documentation`](https://docs.juliadsp.org/stable/periodograms/#Periodograms-periodogram-estimation) for more information.
- window : An optional window function to be applied to the original signal before computing the Fourier transform. Options are [`nothing`, `hamming`, `hanning`]. See the [`DSP documentation`](https://docs.juliadsp.org/stable/periodograms/#Periodograms-periodogram-estimation) for more information.
- kwargs... : All other keyword arguments are passed to Makie to control figure properties.

Note: this function requires Makie to be loaded.

See also [`powerspectrumplot`](@ref), [`powerspectrum`](@ref).
"""
function powerspectrumplot! end

function adjacency end
function adjacency! end

macro wiring_rule(sig, body)
    @gensym g
    fdef = @match sig begin
        :(($src, $dst; $(kwargs...))) => :(function $GraphDynamics.system_wiring_rule!($g, $src, $dst; $(kwargs...))
                                               @graph! $g $body
                                           end)
        :(($src, $dst; $(kwargs...)) where {$(WhereStuff...)}) =>
            :(function $GraphDynamics.system_wiring_rule!($g, $src, $dst; $(kwargs...)) where {$(WhereStuff...)}
                  @graph! $g $body
              end)
        _ => error()
    end
    fdef = Substitute(ex -> ex === :__graph__)(_ -> g, fdef)
    fdef.args[2].args[1] = __source__
    fdef.args[2].args[2] = __source__
    Expr(:block, __source__, esc(Core.@__doc__ fdef))
end

function __init__()
    if parse(Bool, Preferences.@load_preference("PrintLicense", "true"))
        print_license()
    end
end


export Neuron, AbstractBlox, AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, AbstractComposite, AbstractReceptor, AbstractNeuralMass, AbstractModulator, AbstractDiscrete, AbstractStimulus
export AbstractAgent, AbstractEnvironment, AbstractLearningRule, NoLearningRule, AbstractActionSelection, AbstractExperimentMonitor
export Neuron, AbstractBlox, AbstractNeuron, AbstractExciNeuron, AbstractInhNeuron, AbstractComposite, AbstractReceptor, AbstractNeuralMass, AbstractModulator, AbstractDiscrete, AbstractStimulus, AbstractSimpleStimulus
export AbstractAgent, AbstractEnvironment, AbstractLearningRule, NoLearningRule, AbstractActionSelection
export AdjacencyMatrix
export action_selection_from_graph
export get_neurons, get_exci_neurons, get_inh_neurons, get_neuron_color
export detect_spikes, firing_rate, inter_spike_intervals, flat_inter_spike_intervals, powerspectrum
export voltage_timeseries, meanfield_timeseries, state_timeseries
export nameof, namespaced_name, namespaced_nameof, full_namespaced_nameof
export maybe_set_state_pre, maybe_set_state_post, connect_action_selection!
export weight_gradient, get_trial_stimulus
export run_warmup, run_experiment!, run_trial!, reset!, increment_trial!, update_monitor!
export meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, frplot, frplot!, voltage_stack, adjacency, adjacency!
export powerspectrumplot, powerspectrumplot!, welch_pgram, periodogram, hanning, hamming
export inputs, outputs, output
export get_parts, get_neurons, get_exci_neurons, get_inh_neurons, get_weight
export get_connection_matrix, get_components, get_dynamics_components, get_weightmatrix
export get_neuron_color, get_discrete_parts, get_gap, get_gap_weight, get_learning_rule
export get_density, get_connection_rule, get_event_time
export to_vector, params_with_substrings, states_with_substrings
export get_blox_by_name
export to_vector
export @named, @graph, @graph!, @blox, @connection, @wiring_rule
export param_symbols, state_symbols, input_symbols, output_symbols, computed_property_symbols, computed_property_with_inputs_symbols
export BasicConnection, EventConnection, ReverseConnection, HHConnection_GAP, HHConnection_GAP_Reverse, MultipointConnection
export SymbolicIndexingInterface, variable_symbols, parameter_symbols, setsym, setp, setu, getp, setp, getsym, getu
export hypergeometric_connections!, density_connections!, indegree_constrained_connections!, weight_matrix_connections!
export create_adjacency_edges!
export save_graph, load_graph, save_problem, load_problem, @save_graph
export system_wiring_rule!
export @experiment

end
