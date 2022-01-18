module Neuroblox

using Reexport
@reexport using ModelingToolkit

using Graphs
using MetaGraphs

using LinearAlgebra: I, diagm, Matrix
using AbstractFFTs
using FFTW
using ToeplitzMatrices
using DSP

include("Neurographs.jl")
include("utilities/SpectralTools.jl")
include("tools/spectraltools.jl")
include("measurement_models/fmri.jl")

@parameters t
D = Differential(t)

function NeuralMass(;name, activation="a_tan", ω=0, ζ=0, k=0, h=0, τ=0, H=0, λ=0, r=0)

       sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

       if activation == "a_tan"

              params = @parameters ω=ω ζ=ζ k=k h=h

              eqs = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*atan((jcn)/h)
                     D(y) ~ -(ω^2)*x]
       end

       if activation == "logistic"

              params = @parameters τ=τ H=H λ=λ r=r

              eqs = [D(x) ~ y - ((2/τ)*x),
                     D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]

       end

       return ODESystem(eqs, t, sts, params; name=name)
end

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix)

       sysx = [s.x for s in sys]
       adjx = adj_matrix .* sysx
       
       eqs = []
       for region_num in 1:length(sys)
              push!(eqs, sys[region_num].jcn ~ sum(adjx[region_num]))
       end

       return @named Circuit = ODESystem(eqs, systems = sys)
end

function ODEfromGraph(;name, g::LinearNeuroGraph)
       sys = [ get_prop(g.graph,v,:blox) for v in 1:nv(g.graph)]
       adj = AdjMatrixfromLinearNeuroGraph(g)
       return @named GraphCircuit = LinearConnections(sys=sys,adj_matrix=adj)
end

export NeuralMass, LinearConnections, ODEfromGraph
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph, add_blox!
export PowerSpectrum
export mar2csd, csd2mar
export hemodynamics!, boldsignal

end
