module Neuroblox

using Reexport
@reexport using ModelingToolkit

using Graphs
using MetaGraphs

include("Neurographs.jl")

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

function Connections(;name, sys=sys, adj_matrix=adj_matrix)

        begin
               eqs = []
               for region_num in 1:length(sys)
                      push!(eqs, sys[region_num].jcn ~ sum(adj_matrix[region_num,:]))
               end
        end

        return @named Circuit = ODESystem(eqs, systems = sys)
end

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix)

       begin
              sysx = [s.x for s in sys]
              adjx = adj_matrix * sysx
              eqs = []
              for region_num in 1:length(sys)
                     push!(eqs, sys[region_num].jcn ~ sum(adjx[region_num]))
              end
       end

       return @named Circuit = ODESystem(eqs, systems = sys)
end

export NeuralMass, Connections, LinearConnections
export AbstractNeuroGraph, LinearNeuroGraph, AdjMatrixfromLinearNeuroGraph

end
