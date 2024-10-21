using Neuroblox
using Random
using Distributions
using OrdinaryDiffEq
using Plots

η_dist = Cauchy(0.12, 0.02)
N = 20
w = 1

blox = [IzhikevichNeuronCC(name=Symbol("Izh$i"), η=rand(η_dist), sⱼ=1.2308/N) for i in 1:N]

g = MetaDiGraph()
add_blox!.(Ref(g), blox)

for i ∈ blox
    for j ∈ blox
        add_edge!(g, i => j; weight=w)
        add_edge!(g, j => i; weight=w)
    end
end

@named sys = system_from_graph(g)
prob = ODEProblem(sys, [], (0.0, 800.0))
sol = solve(prob, Tsit5(), saveat=1.0)

plot(sol)