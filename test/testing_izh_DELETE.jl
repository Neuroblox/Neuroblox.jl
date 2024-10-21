using Neuroblox
using Random
using Distributions
using OrdinaryDiffEq
using Plots

η_dist = Cauchy(0.12, 0.02)
N = 500
w = 1

blox = [IzhikevichNeuronCC(name=Symbol("Izh$i"), η=rand(η_dist), sⱼ=1.2308/N) for i in 1:N]

g = MetaDiGraph()
add_blox!.(Ref(g), blox)

for i ∈ 1:N
    for j ∈ 1:N
        add_edge!(g, i, j, Dict(:weight => w))
    end
end

@named sys = system_from_graph(g; graphdynamics=true)
prob = ODEProblem(sys, [], (0.0, 200.0))
@time sol = solve(prob, Tsit5(), saveat=1.0)

plot(meanfield_timeseries(blox, sol))