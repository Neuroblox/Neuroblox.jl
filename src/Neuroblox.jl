module Neuroblox

using Reexport
@reexport using ModelingToolkit

@parameters t
D = Differential(t)

function NeuralMass_Logistic(;name, τ=τ, H=H, λ=λ, r=r)

       sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
       params = @parameters τ=τ H=H λ=λ r=r

       eqs = [D(x) ~ y - ((2/τ)*x),
              D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]

       return ODESystem(eqs, t, sts, params; name=name)
end

function NeuralMass_aTan(;name, ω=ω, ζ=ζ, k=k, h=h)
       
       sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
       params = @parameters ω=ω ζ=ζ k=k h=h

       eqs = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*atan((jcn)/h)
              D(y) ~ -(ω^2)*x]
       
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

export NeuralMass_Logistic, NeuralMass_aTan, Connections

end
