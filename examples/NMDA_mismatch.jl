using Neuroblox
using OrdinaryDiffEq
using Random
using StatsBase
using CairoMakie

@named hh1 = HHNeuronExciBlox(; I_bg=0.4)
@named hh2 = HHNeuronExciBlox(; I_bg=0.35)
@named nmda = MoradiNMDAR(; spk_coeff=1)

g = MetaDiGraph()
add_edge!(g, hh1 => nmda; weight=1)
add_edge!(g, hh2 => nmda; weight=1, reverse=true)
add_edge!(g, nmda => hh2; weight=1)

tspan = (0.0, 5000.0)
@named sys = system_from_graph(g, graphdynamics=false)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(); saveat=0.05)

lines(sol.t, sol[:nmda₊I]; color=:green) # for GD plotting
# lines(sol.t, sol[nmda.I]; color=:green) # for MTK plotting
