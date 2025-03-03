using Neuroblox
using OrdinaryDiffEq
using Random, Distributions
using Plots
#import Neuroblox: AbstractNeuronBlox, paramscoping
#using BenchmarkTools

include("adams_graphdynamics_draft.jl")

ḡᵢ = 0.5
ḡₑ = 0.2

NE = 80
NI = 20

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(1.5, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.1, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        make_nmda_edge!(g, ne, ni)
    end
end

for ne ∈ exci
    for ne ∈ exci
        add_edge!(g, ne => ne; weight=1.0)
    end
end

for ni ∈ inhi
    for ne ∈ exci[1:20]
        add_edge!(g, ni => ne; weight=ḡᵢ/NI)
    end
end

begin
    tspan = (0.0, 1000.0)
    @time sys = system_from_graph(g, graphdynamics=true)
    @time prob = ODEProblem(sys, [], tspan)
    @time sol = solve(prob, Tsit5(), saveat=0.5)
end


### Older tests

## Test network without NMDAR connections
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


### Testing single neuron connections

exci = AdamPYR(name=:PYR, Iₐₚₚ=0.25)
glur = AdamGlu(name=:Glu, θ=-59.0)
nmda = AdamNMDAR(name=:NMDA)
exci2 = AdamPYR(name=:PYR2, Iₐₚₚ=0.33)

g = MetaDiGraph()
add_edge!(g, exci => glur; weight=1.0)
add_edge!(g, glur => nmda; weight=1.0)
add_edge!(g, exci2 => nmda; weight=1.0)
add_edge!(g, nmda => exci2; weight=1.0)

tspan = (0.0, 500.0)
@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=0.5)
plot(sol)

### Testing multiple neuron connections
NE = 800
NI = 800

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(0.25, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.3, 0.05))) for i in 1:NI] # bump up to 0.3
nmdar = [AdamNMDAR(name=Symbol("NMDA$i")) for i in 1:NE]
glu = [AdamGlu(name=Symbol("Glu$i")) for i in 1:NI]

g = MetaDiGraph()

for i in axes(exci, 1)
    add_edge!(g, exci[i] => glu[i]; weight=1.0)
    add_edge!(g, glu[i] => nmdar[i]; weight=1.0)
    add_edge!(g, inhi[i] => nmdar[i]; weight=1.0)
    add_edge!(g, nmdar[i] => inhi[i]; weight=1.0)
end

tspan = (0.0, 500.0)
@time @named sys = system_from_graph(g, graphdynamics=true)
@time prob = ODEProblem(sys, [], tspan)
@time sol = solve(prob, Tsit5(), saveat=0.5)

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