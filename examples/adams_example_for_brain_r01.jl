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

# PING QIF theta-nested gamma oscillations



# Chen/Campbell populations - limited utility for now
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



## DO NOT TOUCH LARGELY FAILURES THAT RUN sol_dde_with_delays
# I want to salvage a bit of this so leaving in for now
# -AGC

# # Reproducing ketamine dynamics
# @named PYR = PYR_Izh(η̄=0.08, κ=0.5, I_ext=0.25)
# @named INP = PYR_Izh(η̄=0.08, κ=0.5, wⱼ=0.0095, a=0.077, I_ext=0.5)

# g = MetaDiGraph()
# add_blox!.(Ref(g), [PYR, INP])
# add_edge!(g, PYR => INP; weight=1.0) #weight is acutaally meaningless here
# add_edge!(g, INP => PYR; weight=1.0) #weight is acutaally meaningless here

# @named sys = system_from_graph(g)
# sys = structural_simplify(sys)

# sim_dur = 3000.0
# prob = ODEProblem(sys, [], (0.0, sim_dur))
# sol = solve(prob, Tsit5(), saveat=1.0)

# # Reproduce Chen/Campbell figure 3 panels on the right, especially 3f
# using Plots
# plot(sol, idxs=[1, 5])
# plot(sol, idxs=[2, 6])
# plot(sol, idxs=[3, 7])

# # Reproducing ketamine dynamics
# kappa_mod = 0.9
# I_adj = 0.1
# ω=4.0
# @named PYR = PYR_Izh(η̄=0.08, κ=kappa_mod, I_ext=I_adj, ω=ω*2*π/1000)
# @named INP = PYR_Izh(η̄=0.08, κ=1-kappa_mod, wⱼ=0.0095*15, a=0.077, τₛ=5.0, I_ext=I_adj, ω=ω*2*π/1000)

# g = MetaDiGraph()
# add_blox!.(Ref(g), [PYR, INP])
# add_edge!(g, PYR => INP; weight=1.0) 
# add_edge!(g, INP => PYR; weight=1.0) 

# @named sys = system_from_graph(g)
# sys = structural_simplify(sys)

# sim_dur = 1000000.0
# prob = ODEProblem(sys, [], (0.0, sim_dur))
# sol = solve(prob, Tsit5(), saveat=0.1)

# # Reproduce Chen/Campbell figure 3 panels on the right, especially 3f
# #using Plots, DSP
# plot(sol, idxs=[1, 5])
# plot(sol, idxs=[2, 6])

# data = Array(sol)
# data = data .+ rand(Normal(0, 0.1), size(data))
# wp = welch_pgram(data[6, :]; fs=10000)
# plot(wp.freq, pow2db.(wp.power), xlim=(0, 50))

# spec = spectrogram(data[6, :], 200; fs=10000)
# heatmap(spec.time, spec.freq, pow2db.(spec.power), ylim=(0, 50), xlim=(0, 100))