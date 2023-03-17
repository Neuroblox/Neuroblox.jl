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
using OrdinaryDiffEq, DataFrames
using Interpolations, DataInterpolations
import Distributions
using Random

# define abstract types for Neuroblox
abstract type Blox end
abstract type BloxConnection end
abstract type BloxUtilities end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type NeuronBlox <: Blox end
abstract type NeuralMassBlox <: Blox end
# abstract type SourceBlox <: Blox end will be added later

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

include("Neurographs.jl")
include("utilities/spectral_tools.jl")
include("utilities/learning_tools.jl")
include("utilities/helperfunctions.jl")
include("control/controlerror.jl")
include("measurement_models/fmri.jl")
include("functional_connectivity_estimators/spectralDCM.jl")
include("blox/neural_mass.jl")
include("blox/cortical_blox.jl")
include("blox/canonicalmicrocircuit.jl")
include("blox/neuron_models.jl")
include("blox/synaptic_network.jl")
include("blox/van_der_pol.jl")
include("blox/ts_outputs.jl")
include("blox/sources.jl")
include("blox/rl_blox.jl")
include("gui/GUI.jl")


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

export harmonic_oscillator, jansen_ritC, jansen_ritSC, jansen_rit_spm12, cmc, cmc_singleregion, next_generation, thetaneuron, qif_neuron, if_neuron, synaptic_network, van_der_pol, wilson_cowan
export IFNeuronBlox, LIFneuron, QIFNeuronBlox, HHNeuronExciBlox, HHNeuronInhibBlox, WilsonCowanBlox, HarmonicOscillatorBlox, JansenRitCBlox, JansenRitSCBlox, LarterBreakspearBlox, CorticalBlox, CorticalBloxNew
export LearningBlox
export CosineSource, CosineBlox, NoisyCosineBlox, PhaseBlox
export PowerSpectrumBlox, BandPassFilterBlox
export phase_inter, phase_sin_blox, phase_cos_blox
export LinearConnections, ODEfromGraph, ODEfromGraphdirect, ODEfromGraphdirect_tmp, connectcomplexblox, spikeconnections, AdjMatrixfromLinearNeuroGraph, adjmatrixfromdigraph, create_rl_loop
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph, add_blox!
export powerspectrum, complexwavelet, bandpassfilter, hilberttransform, phaseangle, mar2csd, csd2mar, mar_ml
export learningrate, ControlError
export sigmoid
export hemodynamics!, boldsignal
export variationalbayes
export simulate, random_initials

end