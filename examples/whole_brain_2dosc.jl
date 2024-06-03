# this is an example that reproduces the Virtual Brain tutorial on Resting State networks
using Neuroblox
using CSV
using DataFrames
using MetaGraphs
using DifferentialEquations

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
