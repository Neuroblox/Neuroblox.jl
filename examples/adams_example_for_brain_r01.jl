using Neuroblox
using DifferentialEquations
using DataFrames
using Test
using Distributions
using Statistics
using LinearAlgebra
using Graphs
using MetaGraphs
using Random

@named popP = PYR_Izh(η̄=0.08, κ=0.8)
@named popQ = PYR_Izh(η̄=0.08, κ=0.2, wⱼ=0.0095, a=0.077)

g = MetaDiGraph()
add_blox!.(Ref(g), [popP, popQ])
add_edge!(g, popP => popQ; weight=1.0) #weight is acutaally meaningless here
add_edge!(g, popQ => popP; weight=1.0) #weight is acutaally meaningless here

@named sys = system_from_graph(g)
sys = structural_simplify(sys)

sim_dur = 800.0
prob = ODEProblem(sys, [], (0.0, sim_dur))
sol = solve(prob, Tsit5(), saveat=1.0)

# Reproduce Chen/Campbell figure 3 panels on the right, especially 3f
using Plots
plot(sol, idxs=[1, 5])
plot(sol, idxs=[2, 6])
plot(sol, idxs=[3, 7])