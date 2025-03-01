using Neuroblox
using OrdinaryDiffEq
using Random, Distributions
using Plots
import Neuroblox: AbstractNeuronBlox, paramscoping
#using BenchmarkTools

ḡᵢ = 0.5
ḡₑ = 0.2

NI = 20
NE = 80

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(0.25, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.3, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        add_edge!(g, ne => ni; weight=ḡₑ/NE)
        add_edge!(g, ni => ne; weight=ḡᵢ/NI)
    end
end

tspan = (0.0, 500.0)
# begin
#     @btime @named sys = system_from_graph(g, graphdynamics=true)
#     @btime prob = ODEProblem(sys, [], tspan)
#     @btime sol = solve(prob, Tsit5(), saveat=0.5)
# end

@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=0.5)

plot(sol, idxs=1:5:(NE+NI)*5)



## Testing Glu for threshold setting
## Commented out for now but useful for tuning later so leaving in the file
# exci = AdamPYR(name=:PYR, Iₐₚₚ=0.25)
# glur = AdamGlu(name=:Glu, θ=-59.0)

# g = MetaDiGraph()
# add_edge!(g, exci => glur; weight=1.0)

# tspan = (0.0, 500.0)
# @named sys = system_from_graph(g, graphdynamics=false)
# prob = ODEProblem(sys, [], tspan)
# sol = solve(prob, Tsit5(), saveat=0.5)
# plot(sol, idxs=6)