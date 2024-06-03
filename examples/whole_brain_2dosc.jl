# this is an example that reproduces the Virtual Brain tutorial on Resting State networks
using Neuroblox
using CSV
using DataFrames
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics

weights = CSV.read("examples/weights.csv",DataFrame)
region_names = names(weights)

wm = Array(weights)

blocks = []
for i in 1:size(wm)[1]
    push!(blocks, Neuroblox.Generic2dOscillator(name=Symbol(region_names[i])))
end

g = MetaDiGraph()
add_blox!.(Ref(g), blocks)
create_adjacency_edges!(g, wm)

@named sys = system_from_graph(g)
sys = structural_simplify(sys)

prob = SDEProblem(sys,rand(-2:0.1:4,76*2), (0.0, 10*60e3), [])
@time sol = solve(prob, EulerHeun(), dt=0.5, saveat=5)

plot(sol.t,sol[5,:],xlims=(0,10000))

solv = Array(sol[1:2:end,1000:2000])'
cor_matrix = cor(solv)
heatmap(atanh.(cor_matrix))
