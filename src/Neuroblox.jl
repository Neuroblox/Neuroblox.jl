module Neuroblox

import Base: merge

using Base.Threads: nthreads

using OhMyThreads: tmapreduce

using Reexport
@reexport using ModelingToolkit
@reexport using ModelingToolkit: ModelingToolkit.t_nounits as t, ModelingToolkit.D_nounits as D

@reexport using ModelingToolkitStandardLibrary.Blocks
@reexport import Graphs: add_edge!
@reexport using MetaGraphs: MetaDiGraph

using Graphs
using MetaGraphs

using ForwardDiff: Dual, Partials, jacobian
using ForwardDiff
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true
using ChainRules: _eigen_norm_phase_fwd!

using LinearAlgebra
using ToeplitzMatrices: Toeplitz
using ExponentialUtilities: exponential!

using DSP, Statistics

using Interpolations
using Random
using OrderedCollections

using StatsBase: sample
using Distributions

using SciMLBase: SciMLBase, AbstractSolution, solve, remake

using ModelingToolkit: get_namespace, get_systems, isparameter,
                    renamespace, namespace_equation, namespace_parameters, namespace_expr,
                    AbstractODESystem, VariableTunable, getp
import ModelingToolkit: equations, inputs, outputs, unknowns, parameters, discrete_events, nameof, getdescription

using Symbolics: @register_symbolic, getdefaultval, get_variables

using CSV: read, write
using DataFrames

using Peaks: argmaxima, peakproms!, peakheights!, findmaxima
using SparseArrays

using LogExpFunctions: logistic

# define abstract types for Neuroblox
abstract type AbstractBlox end # Blox is the abstract type for Blox that are displayed in the GUI
abstract type AbstractComponent end
abstract type BloxConnection end
abstract type BloxUtilities end
abstract type Merger end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type AbstractNeuronBlox <: AbstractBlox end
abstract type NeuralMassBlox <: AbstractBlox end
abstract type CompositeBlox <: AbstractBlox end
abstract type StimulusBlox <: AbstractBlox end
abstract type ObserverBlox end # not AbstractBlox since it should not show up in the GUI
abstract type AbstractPINGNeuron <: AbstractNeuronBlox end

# we define these in neural_mass.jl
# abstract type HarmonicOscillatorBlox <: NeuralMassBlox end
# abstract type JansenRitCBlox <: NeuralMassBlox end
# abstract type JansenRitSCBlox <: NeuralMassBlox end
# abstract type WilsonCowanBlox <: NeuralMassBlox end

# abstract type DynamicSignalBlox <: Blox end
# abstract type PhaseSignalBlox <: DynamicSignalBlox end
# abstract type TSfromPSDBlox <: DynamicSignalBlox end

abstract type SpectralUtilities <: BloxUtilities end 

# abstract type MathBlox <: Blox end
# abstract type FilterBlox <: Blox end
# abstract type ControlBlox <: Blox end

abstract type BloxConnectFloat <: BloxConnection end
abstract type BloxConnectComplex <: BloxConnection end
abstract type BloxConnectMultiFloat <: BloxConnection end
abstract type BloxConnectMultiComplex <: BloxConnection end

# dictionary type for Blox parameters
Para_dict = Dict{Symbol, Union{<: Real, Num}}

include("utilities/spectral_tools.jl")
include("utilities/learning_tools.jl")
include("utilities/bold_methods.jl")
include("control/controlerror.jl")
include("measurementmodels/fmri.jl")
include("measurementmodels/lfp.jl")
include("datafitting/spDCM_VL.jl")
include("blox/neural_mass.jl")
include("blox/cortical.jl")
include("blox/canonicalmicrocircuit.jl")
include("blox/neuron_models.jl")
include("blox/DBS_Model_Blox_Adam_Brown.jl")
include("blox/ts_outputs.jl")
include("blox/sources.jl")
include("blox/DBS_sources.jl")
include("blox/rl_blox.jl")
include("blox/winnertakeall.jl")
include("blox/subcortical_blox.jl")
include("blox/stochastic.jl")
include("blox/discrete.jl")
include("blox/ping_neuron_examples.jl")
include("blox/reinforcement_learning.jl")
include("gui/GUI.jl")
include("blox/connections.jl")
include("blox/blox_utilities.jl")
include("GraphDynamicsInterop/GraphDynamicsInterop.jl")
include("Neurographs.jl")
include("adjacency.jl")

const Neuron = AbstractNeuronBlox
const SpikeSource = AbstractSpikeSource

function simulate(sys::ODESystem, u0, timespan, p, solver = AutoVern7(Rodas4()); kwargs...)
    prob = ODEProblem(sys, u0, timespan, p)
    sol = solve(prob, solver; kwargs...) #pass keyword arguments to solver
    return DataFrame(sol)
end

function simulate(blox::CorticalBlox, u0, timespan, p, solver = AutoVern7(Rodas4()); kwargs...)
    prob = ODEProblem(blox.system, u0, timespan, p)
    sol = solve(prob, solver; kwargs...) # pass keyword arguments to solver
    statesV = [s for s in unknowns(blox.system) if contains(string(s),"V")]
    vsol = sol[statesV]
    vmean = vec(mean(hcat(vsol...),dims=2))
    df = DataFrame(sol)
    vlist = Symbol.(statesV)
    pushfirst!(vlist,:timestamp)
    dfv = df[!,vlist]
    dfv[!,:Vmean] = vmean
    return dfv
end

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
function random_initials(odesys::ODESystem, blox)
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

function meanfield end
function meanfield! end

function rasterplot end
function rasterplot! end

function stackplot end
function stackplot! end

function frplot end
function frplot! end

function voltage_stack end

function ecbarplot end
function ecbarplot! end

function freeenergy end
function freeenergy! end

function powerspectrumplot end
function powerspectrumplot! end

function adjacency end
function adjacency! end

function __init__()
    #if Preferences.@load_preference("PrintLicense", true)
        print_license()
    #end
end


export Neuron
export JansenRitSPM12, qif_neuron, if_neuron, hh_neuron_excitatory, 
    hh_neuron_inhibitory, VanDerPol, Generic2dOscillator, kuramoto_oscillator
export HHNeuronExciBlox, HHNeuronInhibBlox, IFNeuron, LIFNeuron, QIFNeuron, IzhikevichNeuron, LIFExciNeuron, LIFInhNeuron,
    CanonicalMicroCircuitBlox, WinnerTakeAllBlox, CorticalBlox, SuperCortical, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox,
    HHNeuronInhib_GPe_Adam_Blox, Striatum_MSN_Adam, Striatum_FSI_Adam, GPe_Adam, STN_Adam, LIFExciCircuitBlox, LIFInhCircuitBlox
export LinearNeuralMass, HarmonicOscillator, JansenRit, WilsonCowan, LarterBreakspear, KuramotoOscillator
export Matrisome, Striosome, Striatum, GPi, GPe, Thalamus, STN, TAN, SNc
export HebbianPlasticity, HebbianModulationPlasticity
export Agent, ClassificationEnvironment, GreedyPolicy, reset!
export LearningBlox
export CosineSource, CosineBlox, NoisyCosineBlox, PhaseBlox, ImageStimulus, ConstantInput, ExternalInput, SpikeSource, PoissonSpikeTrain, generate_spike_times
export DBS, ProtocolDBS, detect_transitions, compute_transition_times, compute_transition_values, get_protocol_duration
export BandPassFilterBlox
export OUBlox, OUCouplingBlox, ARBlox
export phase_inter, phase_sin_blox, phase_cos_blox
export SynapticConnections, create_rl_loop
export add_blox!, get_system
export powerspectrum, complexwavelet, bandpassfilter, hilberttransform, phaseangle, mar2csd, csd2mar, mar_ml
export learningrate, ControlError
export vecparam, csd_Q, setup_sDCM, run_sDCM_iteration!, defaultprior
export simulate, random_initials
export system_from_graph, system, graph_delays
export create_adjacency_edges!, adjmatrixfromdigraph
export get_namespaced_sys, nameof
export run_experiment!, run_trial!
export addnontunableparams, changetune
export get_weights, get_dynamic_states, get_idx_tagged_vars, get_eqidx_tagged_vars
export BalloonModel,LeadField, boldsignal_endo_balloon
export PINGNeuronExci, PINGNeuronInhib
export NGNMM_Izh, NGNMM_QIF, NGNMM_theta, NextGenerationEIBlox
export meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, frplot, frplot!, voltage_stack, ecbarplot, ecbarplot!, freeenergy, freeenergy!, adjacency, adjacency!
export powerspectrumplot, powerspectrumplot!, welch_pgram, periodogram, hanning, hamming
export detect_spikes, mean_firing_rate, firing_rate
export voltage_timeseries, meanfield_timeseries, state_timeseries, get_neurons, get_exci_neurons, get_inh_neurons, get_neuron_color
export AdjacencyMatrix, Connector, connection_rule, connection_equations, connection_spike_affects, connection_learning_rules, connection_callbacks
export inputs, outputs, equations, unknowns, parameters, discrete_events
export MetabolicHHNeuron
end
