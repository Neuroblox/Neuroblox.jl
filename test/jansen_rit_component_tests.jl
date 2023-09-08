using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

# test blox
@named PY  = JansenRitCBloxDelay(τ=0.001, H=20, λ=5, r=0.15)
@named EI  = JansenRitCBloxDelay(τ=0.01, H=20, λ=5, r=5)
@named II  = JansenRitCBloxDelay(τ=2.0, H=60, λ=5, r=5)

# test graphs
g = MetaDiGraph()
add_blox!(g, PY)
add_blox!(g, EI)
add_blox!(g, II)
add_edge!(g, 1, 2, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 2, 3, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 3, 1, Dict(:weight => 1.0, :delay => 1.5))

@named final_system = system_from_graph(g)
sim_dur = 10.0 # Simulate for 10 Seconds
sys = structural_simplify(final_system)
prob = DDEProblem(sys,
    [],
    (0.0, sim_dur),
    constant_lags = [1])
alg = MethodOfSteps(Tsit5())
sol_mtk = solve(prob, alg, reltol = 1e-7, abstol = 1e-10, saveat=0.001)