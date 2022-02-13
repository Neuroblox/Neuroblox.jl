module Neuroblox

using Reexport
@reexport using ModelingToolkit

using Graphs
using MetaGraphs

import LinearAlgebra as la
using AbstractFFTs
using FFTW
import ToeplitzMatrices as tm
using DSP
import ExponentialUtilities as eu
using OrdinaryDiffEq, DataFrames

include("Neurographs.jl")
include("utilities/SpectralTools.jl")
include("measurement_models/fmri.jl")
include("functional_connectivity_estimators/spectralDCM.jl")
include("blox/neuralmass.jl")
include("blox/thetaneuron.jl")
include("blox/qif_neuron.jl")
include("blox/synaptic_network.jl")

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
       adj = adj_matrix .* connector
       eqs = []
       for region_num in 1:length(sys)
              push!(eqs, sys[region_num].jcn ~ sum(adj[region_num]))
       end
       return @named Circuit = ODESystem(eqs, systems = sys)
end

function ODEfromGraph(;name, g::LinearNeuroGraph, connector=connector)
       sys = [ get_prop(g.graph,v,:blox) for v in 1:nv(g.graph)]
       adj = AdjMatrixfromLinearNeuroGraph(g)
       return @named GraphCircuit = LinearConnections(sys=sys,adj_matrix=adj,connector=connector)
end

function simulate(sys::ODESystem, u0, timespan, p, solver = Tsit5())
       prob = ODAEProblem(structural_simplify(sys), u0, timespan, p)
       sol = solve(prob, solver)
       return DataFrame(sol)
end

export neuralmass, thetaneuron
export LinearConnections, ODEfromGraph
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph, add_blox!
export powerspectrum, complexwavelet, mar2csd, csd2mar, mar_ml
export hemodynamics!, boldsignal
export variationalbayes
export simulate

end
