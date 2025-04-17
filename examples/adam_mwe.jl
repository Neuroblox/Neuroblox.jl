using Neuroblox
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using CairoMakie

PYR = AdamPYR(name=Symbol("PYR"), Iₐₚₚ=-0.25*1e-3)
INP = AdamIN(name=Symbol("INP"), Iₐₚₚ=0.1*1e-3)

g = MetaDiGraph()

NMDA = AdamNMDAR(name=Symbol("NMDA"), k_unblock=5.4, g=9.5)
add_edge!(g, PYR => NMDA; weight=1)
add_edge!(g, INP => NMDA; weight=1, reverse=true)
add_edge!(g, NMDA => INP; weight=1)

#=
AMPA = AdamAMPA(name=Symbol("AMPA"))
add_edge!(g, PYR => AMPA; weight=1)
add_edge!(g, AMPA => INP; weight=1)

GABA = AdamGABA(name=Symbol("GABA"), g = 0.8)
add_edge!(g, INP => GABA; weight=1)
add_edge!(g, GABA => PYR; weight=1)
=#

tspan = (0.0, 5000.0)
@named sys = system_from_graph(g, graphdynamics=false)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, RK4(); saveat=0.05)

rasterplot([INP, PYR], sol; threshold=-10)
