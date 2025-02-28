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
@named exci_PING = QIF_PING_NGNMM(I_ext=10.0, ω=5*2*π/1000, J_internal=8.0, H=1.3, Δ=1.0, τₘ=20.0, A=0.2)
@named inhi_PING = QIF_PING_NGNMM(I_ext=5.0, ω=5*2*π/1000, J_internal=0.0, H=-5.0, Δ=1.0, τₘ=10.0, A=0.0)

g = MetaDiGraph()
add_blox!.(Ref(g), [exci_PING, inhi_PING])
add_edge!(g, exci_PING => inhi_PING; weight=10.0)
add_edge!(g, inhi_PING => exci_PING; weight=10.0)

@named sys = system_from_graph(g)

sim_dur = 1000.0
prob = SDEProblem(sys, [], (0.0, sim_dur))
sol = solve(prob, RKMil(), saveat=0.1)

using Plots, DSP, CSV
plot(sol, idxs=[1, 3])
plot(sol, idxs=[2, 4])

plot(sol, idxs=[1, 3], labels=["Excitatory Population Firing Rate" "Inhibitory Population Firing Rate"], xlabel="Time (ms)", ylabel="Firing Rate (a.u.)", title="PING QIF theta-nested gamma oscillations")
plot(sol, idxs=[2, 4], labels=["Excitatory Population Membrane Potential" "Inhibitory Population Membrane Potential"], xlabel="Time (ms)", ylabel="Membrane Potential (a.u))", title="PING QIF theta-nested gamma oscillations")

data = Array(sol)
#data = data .+ rand(Normal(0, 10), size(data))
wp = welch_pgram(data[2, :]; fs=2000)
plot(wp.freq, pow2db.(wp.power), xlim=(0, 50), xlabel="Frequency (Hz)" , ylabel="Power (dB)", title="Power Spectrum of Excitatory Population")

x = 0:0.1:1000.0
y = sin.(2*π*5*x/1000)
plot(x, y, label="NMDAR driving current", xlabel="Time (ms)", ylabel="Current (a.u.)", title="NMDAR driving current")

using CairoMakie
df = DataFrame(sol)
f = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (900, 1000), fontsize=19)
ga = f[1, 1]
gb = f[2, 1]
gc = f[3, 1]

axa = Axis(ga, ylabel = "Firing Rate (a.u.)", title="Population Firing Rate", xticklabelsvisible=false, ylabelsize=23, titlesize=25)
axb = Axis(gb, ylabel = "Membrane Potential (a.u.)", title="Population Membrane Voltage", xticklabelsvisible=false, ylabelsize=23, titlesize=25)
axc = Axis(gc, ylabel = "Current (a.u.)", xlabel="Time (ms)", title="NMDA driving current", ylabelsize=23, xlabelsize=23, titlesize=25)

lines!(axa, df.timestamp, df.exci_PING₊r, color = :blue, linewidth = 1.5, label = "Excitatory")
lines!(axa, df.timestamp, df.inhi_PING₊r, color = :red, linewidth = 1.5, label = "Inhibitory")
lines!(axb, df.timestamp, df.exci_PING₊v, color = :blue, linewidth = 1.5, label = "Excitatory")
lines!(axb, df.timestamp, df.inhi_PING₊v, color = :red, linewidth = 1.5, label = "Inhibitory")
lines!(axc, x, y, color = :green, linewidth = 1.5, label = "NMDAR driving current")
rowsize!(f.layout, 1, Auto(1.3))
rowsize!(f.layout, 2, Auto(1.3))
rowsize!(f.layout, 3, Auto(0.6))
axislegend(axa, position = :rt)
axislegend(axb, position = :rt)

f

# Chen/Campbell populations - limited utility for now
# @named popP = PYR_Izh(η̄=0.08, κ=0.8)
# @named popQ = PYR_Izh(η̄=0.08, κ=0.2, wⱼ=0.0095, a=0.077)

# g = MetaDiGraph()
# add_blox!.(Ref(g), [popP, popQ])
# add_edge!(g, popP => popQ; weight=0.8) #weight is acutally meaningless here
# add_edge!(g, popQ => popP; weight=0.2) #weight is acutally meaningless here

# @named sys = system_from_graph(g)

# sim_dur = 800.0
# prob = ODEProblem(sys, [], (0.0, sim_dur))
# sol = solve(prob, Tsit5(), saveat=1.0)

# # Reproduce Chen/Campbell figure 3 panels on the right, especially 3f
# using Plots
# plot(sol, idxs=[1, 5])
# plot(sol, idxs=[2, 6])
# plot(sol, idxs=[3, 7])



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