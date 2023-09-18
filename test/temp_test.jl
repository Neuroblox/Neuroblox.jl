using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

@named LB1 = LarterBreakspear()
@named LB2 = LarterBreakspear()

adj = [0 1; 1 0]
g = MetaDiGraph()
add_blox_list!(g, [LB1, LB2])
create_adjacency_edges!(g, adj)

@named sys = system_from_graph(g)
sys = structural_simplify(sys)

sim_dur = 1e2
prob = ODEProblem(sys, [], (0.0, sim_dur), [])
sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
@test sol.retcode == ReturnCode.Success