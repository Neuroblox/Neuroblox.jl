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
include("blox/neuron_models.jl")
include("blox/synaptic_network.jl")
include("blox/van_der_pol.jl")

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
       adj = adj_matrix .* connector
       eqs = []
       for region_num in 1:length(sys)
              push!(eqs, sys[region_num].jcn ~ sum(adj[region_num]))
       end
       return @named Circuit = ODESystem(eqs, systems = sys)
end

function ODEfromGraph(;name, g::LinearNeuroGraph)
       sys = [ get_prop(g.graph,v,:blox) for v in 1:nv(g.graph)]
       connector = [s.x for s in sys]
       adj = AdjMatrixfromLinearNeuroGraph(g)
       return @named GraphCircuit = LinearConnections(sys=sys,adj_matrix=adj,connector=connector)
end

function simulate(sys::ODESystem, u0, timespan, p, solver = Tsit5())
       prob = ODAEProblem(sys, u0, timespan, p)
       sol = solve(prob, solver)
       return DataFrame(sol)
end

function simulate_neurons(sys::ODESystem, u0, timespan, p, solver = Rodas5()) #ODESystem is sent after structural_simplify
       prob = ODEProblem(sys, u0, timespan, p)
       solver=solver
       sol = solve(prob,solver,saveat=0.01,reltol=1e-4,abstol=1e-4)
       return DataFrame(sol)
end

function simulate_complex(sys::ODESystem, u0, timespan, p, solver=Tsit5())
       prob = ODEProblem(sys, u0, timespan, p)
       prob = remake(prob, u0=complex.(prob.u0))
       solver=solver
       sol = solve(prob, Tsit5())
       return sol
end

export neuralmass, thetaneuron, qif_neuron, if_neuron, synaptic_network, van_der_pol
export LinearConnections, ODEfromGraph
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph, add_blox!
export powerspectrum, complexwavelet, mar2csd, csd2mar, mar_ml
export hemodynamics!, boldsignal
export variationalbayes
export simulate, simulate_neurons, simulate_complex

end
