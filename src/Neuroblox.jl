module Neuroblox

using Reexport
@reexport using ModelingToolkit
@reexport using ModelingToolkitStandardLibrary.Blocks
@reexport using Graphs: add_edge!

using Graphs
using MetaGraphs

using ForwardDiff: Dual, Partials, jacobian
using ForwardDiff
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true
using ChainRules: _eigen_norm_phase_fwd!

using LinearAlgebra
using ToeplitzMatrices: Toeplitz
using ExponentialUtilities: exponential!

using AbstractFFTs
using FFTW
using DSP, Statistics
using OrdinaryDiffEq
using DifferentialEquations
using Interpolations
using Random
using OrderedCollections
using DelayDiffEq

using StatsBase: sample
using Distributions

using ModelingToolkit: get_namespace, get_systems, isparameter,
                    renamespace, namespace_equation, namespace_parameters, namespace_expr,
                    AbstractODESystem
import ModelingToolkit: inputs, nameof, outputs, getdescription

using Symbolics: @register_symbolic, getdefaultval
using IfElse

using DelimitedFiles: readdlm
using CSV: read
using DataFrames

using Peaks: argmaxima, peakproms!, peakheights!

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

# we define these in neural_mass.jl
# abstract type HarmonicOscillatorBlox <: NeuralMassBlox end
# abstract type JansenRitCBlox <: NeuralMassBlox end
# abstract type JansenRitSCBlox <: NeuralMassBlox end
# abstract type WilsonCowanBlox <: NeuralMassBlox end
# abstract type NextGenerationBlox <: NeuralMassBlox end

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

# struct types for Variational Laplace
mutable struct VLState
    iter::Int
    v::Float64             # log ascent rate
    F::Vector{Float64}
    dF::Vector{Float64}
    λ::Vector{Float64}
    ϵ_θ::Vector{Float64}
    μθ_po::Vector{Float64}
    Σθ_po::Matrix{Float64}
    dFdθ::Vector{Float64}
    dFdθθ::Matrix{Float64}
end

struct VLSetup
    model_at_x0
    y_csd::Array{Complex}
    tolerance::Float64
    systemnums::Vector{Int}
    systemvecs::Vector{Vector{Float64}}
    systemmatrices::Vector{Matrix{Float64}}
    Q::Matrix{Complex}
end


include("utilities/spectral_tools.jl")
include("utilities/learning_tools.jl")
include("utilities/bold_methods.jl")
include("control/controlerror.jl")
include("measurementmodels/fmri.jl")
include("datafitting/spectralDCM.jl")
include("blox/neural_mass.jl")
include("blox/cortical_blox.jl")
include("blox/canonicalmicrocircuit.jl")
include("blox/neuron_models.jl")
include("blox/synaptic_network.jl")
include("blox/van_der_pol.jl")
include("blox/ts_outputs.jl")
include("blox/sources.jl")
include("blox/rl_blox.jl")
include("blox/winnertakeall.jl")
include("blox/subcortical_blox.jl")
include("blox/stochastic.jl")
include("blox/discrete.jl")
include("blox/reinforcement_learning.jl")
include("gui/GUI.jl")
include("blox/connections.jl")
include("blox/blox_utilities.jl")
include("Neurographs.jl")

function simulate(sys::ODESystem, u0, timespan, p, solver = AutoVern7(Rodas4()); kwargs...)
    prob = ODEProblem(sys, u0, timespan, p)
    sol = solve(prob, solver; kwargs...) #pass keyword arguments to solver
    return DataFrame(sol)
end

function simulate(blox::CorticalBlox, u0, timespan, p, solver = AutoVern7(Rodas4()); kwargs...)
    prob = ODEProblem(blox.odesystem, u0, timespan, p)
    sol = solve(prob, solver; kwargs...) #pass keyword arguments to solver
    statesV = [s for s in states(blox.odesystem) if contains(string(s),"V")]
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
    odestates = states(odesys)
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

function __init__()
    #if Preferences.@load_preference("PrintLicense", true)
        print_license()
    #end
end

export JansenRitSPM12, next_generation, qif_neuron, if_neuron, hh_neuron_excitatory, 
    hh_neuron_inhibitory, synaptic_network, van_der_pol
export IFNeuronBlox, LIFNeuronBlox, QIFNeuronBlox, HHNeuronExciBlox, HHNeuronInhibBlox,
    CanonicalMicroCircuitBlox, WinnerTakeAllBlox, CorticalBlox, SuperCortical
export LinearNeuralMass, HarmonicOscillator, JansenRit, WilsonCowan, LarterBreakspear, NextGenerationBlox, NextGenerationResolvedBlox, NextGenerationEIBlox
export Matrisome, Striosome, Striatum, GPi, GPe, Thalamus, STN, TAN, SNc
export HebbianPlasticity, HebbianModulationPlasticity
export Agent, ClassificationEnvironment, GreedyPolicy
export LearningBlox
export CosineSource, CosineBlox, NoisyCosineBlox, PhaseBlox, ImageStimulus
export PowerSpectrumBlox, BandPassFilterBlox
export OUBlox, OUCouplingBlox
export phase_inter, phase_sin_blox, phase_cos_blox
export SynapticConnections, create_rl_loop
export add_blox!
export powerspectrum, complexwavelet, bandpassfilter, hilberttransform, phaseangle, mar2csd, csd2mar, mar_ml
export learningrate, ControlError
export boldsignal, BalloonModel
export vecparam, csd_Q, setup_sDCM, run_sDCM_iteration!
export simulate, random_initials
export system_from_graph, graph_delays
export create_adjacency_edges!, adjmatrixfromdigraph
export get_namespaced_sys, nameof
export run_experiment!, run_trial!
export addnontunableparams, get_hemodynamic_observers
export get_weights
export boldsignal_endo_balloon

end
