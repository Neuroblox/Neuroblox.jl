using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

@named osc1 = HarmonicOscillator()
@named osc2 = HarmonicOscillator()

adj = [0 1; 1 0]
g = MetaDiGraph()
add_blox_list!(g, [osc1, osc2])
create_adjacency_edges!(g, adj)

@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [0.2, 0.4, 0.6, 0.8], (0.0, 5e2), [])
sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
@test sol.retcode == ReturnCode.Success