using Neuroblox
using Random
using Distributions
using OrdinaryDiffEq
using Plots
using OhMyThreads

η_dist = Cauchy(0.12, 0.02)
N = 1000
w = 1
sparsity = 1.0

blox = [IzhikevichNeuronCC(name=Symbol("Izh$i"), η=rand(η_dist), sⱼ=1.2308/N) for i in 1:N]

g = MetaDiGraph()
add_blox!.((g,), blox)
for i ∈ 1:N
    for j ∈ 1:N
        if rand() <= sparsity
            add_edge!(g, i, j, Dict(:weight => w, :connection_rule => "basic"))
        end
    end
end

@named sys = system_from_graph(g; graphdynamics=true)
prob = ODEProblem(sys, [], (0.0, 40.0))
@time sol = solve(prob, Tsit5())

plot(meanfield_timeseries(blox, sol))


### Additional bnechmarking etc...
### Ignore for now
# using Neuroblox, Random, Distributions, OrdinaryDiffEq

# η_dist = Cauchy(0.12, 0.02)
# N = 100
# w = 1

# blo