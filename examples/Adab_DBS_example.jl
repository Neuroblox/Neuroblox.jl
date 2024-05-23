using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

global_ns = :g
@named msn = Striatum_MSN_Adam(namespace=global_ns)
@named fsi = Striatum_FSI_Adam(namespace=global_ns)
@named gpe = GPe_Adam(namespace=global_ns)
@named stn = STN_Adam(namespace=global_ns)

assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)

add_edge!(g, 1, 3, Dict(:weight=> 2.5/100, :density=>0.33))
add_edge!(g, 2, 1, Dict(:weight=> 0.6/50, :density=>0.15))
add_edge!(g, 3, 4, Dict(:weight=> 0.3/80, :density=>0.05))
add_edge!(g, 4, 2, Dict(:weight=> 0.165/40, :density=>0.1))

@named neuron_net = system_from_graph(g)
sys = structural_simplify(neuron_net)
prob = SDEProblem(sys, [], (0.0, 500), [])
sol = solve(prob, saveat = 0.01)
ss=convert(Array,sol)