module Neuroblox

using Reexport
@reexport using ModelingToolkit

using Graphs
using MetaGraphs

import LinearAlgebra as la
using AbstractFFTs
using FFTW
import ToeplitzMatrices as tm
using DSP, Statistics
import ExponentialUtilities as eu
using OrdinaryDiffEq, DataFrames
using Interpolations

# define abstract types for Neuroblox
abstract type Blox end
abstract type BloxConnection end
abstract type BloxUtilities end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type NeuronBlox <: Blox end
abstract type NeuralMassBlox <: Blox end

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
include("blox/canonicalmicrocircuit.jl")
include("blox/theta_neuron.jl")
include("blox/neuron_models.jl")
include("blox/synaptic_network.jl")
include("blox/van_der_pol.jl")
include("blox/ts_outputs.jl")
include("gui/GUI.jl")


function simulate(sys::ODESystem, u0, timespan, p, solver = AutoVern7(Rodas4()); kwargs...)
    prob = ODEProblem(sys, u0, timespan, p)
    sol = solve(prob, solver; kwargs...) #pass keyword arguments to solver
    return DataFrame(sol)
end

export harmonic_oscillator, jansen_ritC, jansen_ritSC, jansen_rit_spm12, cmc, cmc_singleregion, next_generation, thetaneuron, qif_neuron, if_neuron, hh_neuron_excitatory, hh_neuron_inhibitory, synaptic_network, van_der_pol, wilson_cowan
export IFNeuronBlox, QIFNeuronBlox, WilsonCowanBlox, HarmonicOscillatorBlox, JansenRitCBlox, JansenRitSCBlox, LauterBreakspearBlox
export PowerSpectrumBlox, BandPassFilterBlox
export phase_inter, phase_sin_blox, phase_cos_blox
export LinearConnections, ODEfromGraph, connectcomplexblox, AdjMatrixfromLinearNeuroGraph, adjmatrixfromdigraph
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph, add_blox!
export powerspectrum, complexwavelet, bandpassfilter, hilberttransform, phaseangle, mar2csd, csd2mar, mar_ml
export learningrate, ARVTarget, PhaseTarget, ControlError
export sigmoid
export hemodynamics!, boldsignal
export variationalbayes
export simulate

end