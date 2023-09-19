module Neuroblox

using Reexport
@reexport using ModelingToolkit
@reexport using ModelingToolkitStandardLibrary.Blocks

using Graphs
using MetaGraphs

import LinearAlgebra as la
using AbstractFFTs
using FFTW
import ToeplitzMatrices as tm
using DSP, Statistics
import ExponentialUtilities as eu
using OrdinaryDiffEq
using Interpolations
using DataInterpolations
using Distributions
using Random
using OrderedCollections
using DelayDiffEq

using StatsBase: sample

using ModelingToolkit: get_namespace, get_systems, renamespace, 
                    namespace_equation, namespace_variables, namespace_expr,
                    AbstractODESystem
import ModelingToolkit: inputs, nameof

using ModelingToolkitStandardLibrary.Blocks: SampledData

using Symbolics: @register_symbolic
using IfElse

using DelimitedFiles: readdlm
using CSV: read
using DataFrames

# define abstract types for Neuroblox
abstract type AbstractBlox end # Blox is the abstract type for Blox that are displayed in the GUI
abstract type AbstractComponent end
abstract type BloxConnection end
abstract type BloxUtilities end
abstract type Merger end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type AbstractNeuronBlox <: AbstractBlox end
abstract type NeuralMassBlox <: AbstractBlox end
abstract type SuperBlox <: AbstractBlox end
abstract type StimulusBlox <: AbstractBlox end

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

include("utilities/spectral_tools.jl")
include("utilities/learning_tools.jl")
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
include("blox/stochastic.jl")
include("gui/GUI.jl")
include("blox/blox_utilities.jl")
include("blox/connections.jl")
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

export harmonic_oscillator, jansen_ritC, jansen_ritSC, jansen_rit_spm12, 
    next_generation, thetaneuron, qif_neuron, if_neuron, hh_neuron_excitatory, 
    hh_neuron_inhibitory, synaptic_network, van_der_pol, wilson_cowan
export IFNeuronBlox, LIFNeuronBlox, QIFNeuronBlox, HHNeuronExciBlox, HHNeuronInhibBlox, LinearNeuralMassBlox,
    WilsonCowanBlox, HarmonicOscillatorBlox, JansenRitCBlox, JansenRitSCBlox, LarterBreakspearBlox,
    CanonicalMicroCircuitBlox, WinnerTakeAllBlox, CorticalBlox, SuperCortical
export LinearNeuralMass, HarmonicOscillator, JansenRit, WilsonCowan, LarterBreakspear
export LearningBlox
export CosineSource, CosineBlox, NoisyCosineBlox, PhaseBlox, ImageStimulus
export PowerSpectrumBlox, BandPassFilterBlox, PhaseAngleBlox
export OUBlox, OUCouplingBlox
export phase_inter, phase_sin_blox, phase_cos_blox
export LinearConnections, SynapticConnections, ODEfromGraph, ODEfromGraphNeuron, 
    connectcomplexblox, spikeconnections, adjmatrixfromdigraph, create_rl_loop
export add_blox!
export powerspectrum, complexwavelet, bandpassfilter, hilberttransform, phaseangle, mar2csd, csd2mar, mar_ml
export learningrate, ControlError
export Hemodynamics, LinHemo, boldsignal
export vecparam, unvecparam, csd_Q, spectralVI
export simulate, random_initials
export system_from_graph, graph_delays
export create_adjacency_edges!, add_blox_list!
export get_namespaced_sys, namespace_expr, nameof
end