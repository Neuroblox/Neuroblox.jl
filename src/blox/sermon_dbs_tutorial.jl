using Neuroblox
using DifferentialEquations
#using DataFrames
using Distributions, Random
#using Statistics
#using LinearAlgebra
using Graphs, MetaGraphs
using DSP

# First, create the three neurotransmitter pools. As they have the same structure,
# we can use the same constructor with different parameters to create the systems.
# Parameters are taken from the supplementary material Table A.
@named RRP = SermonNPool(τ_pool=1.0, p_pool=0.3)
@named RP  = SermonNPool(τ_pool=10.0, p_pool=0.02)
@named RtP = SermonNPool(τ_pool=50.0, p_pool=0.00035)

# Next, create the DBS stimulator block. This block is a simple pulse with a given width
# and period. To try and recreate Fig. 2A, start stimulation at 50s.
@named DBS = SermonDBS(start_time=50.0, pulse_width=0.0001)

# Next, create a collection of 50 Kuramotor oscillators with noise.
# Set the number of oscillators (default from the paper)
N = 50

# Define the natural distribution of oscillator frequencies
# Parameters from supplementary material Table A
Ω = 249
σ = 26.317

ks_blocks = [KuramotoOscillator(name=Symbol("KO$i"), 
                                ω=rand(Normal(Ω, σ)),
                                ζ=5.920,
                                include_noise=true) for i in 1:N]

# Create a graph and add all the blocks as nodes
g = MetaDiGraph()
add_blox!.(Ref(g), vcat(ks_blocks, DBS, RRP, RP, RtP))

# Additional connection parameters specified from supplementary material Table A
p = @parameters M_RRP=1.0 M_RP=0.5 M_RtP = 0.3 k_μ = 11976.0 I =72053.0

# Connect all oscillators to each other
for i in 1:N
    for j in 1:N
        # Weight is k_μ to be multiplied by the other terms in equation 6
        add_edge!(g, i, j, Dict(:weight => k_μ, :sermon_rule => true, :extra_bloxs => [RRP, RP, RtP], :extra_params => p))
    end
end

# Connect DBS stimulator to oscillators
for i in 1:N
    add_edge!(g, N+1, i, Dict(:weight => I))
end

# Connect DBS stimulator to neurotransmitter pools
add_edge!(g, N+1, N+2, Dict(:weight => 1.0))
add_edge!(g, N+1, N+3, Dict(:weight => 1.0))
add_edge!(g, N+1, N+4, Dict(:weight => 1.0))

@named sys = system_from_graph(g, p)
sys = structural_simplify(sys)

# Simulate for the length of Fig. 2A
sim_len = 150.0
@time prob = SDEProblem(sys, [], (0.0, sim_len))
# EulerHeun because you need to force timestops at the pulse width of the DBS stimulator
# so fixed timestep solver is the easiest way to do this. Maxiter problems can happen with 
# the adaptive ones I've tried.
@time sol = solve(prob, EulerHeun(), dt=0.0001, saveat=0.001)

# Helper function because θ from the Kuramoto oscillators is unbounded and needs wrapping
# to get the correct spectrogram
function wrapTo2Pi(x)
    posinput = x .> 0
    wrapped = mod.(x, 2*π)
    wrapped[wrapped .== 0 .&& posinput] .= 2*π
    return wrapped
end

# Helper code to plot the spectrogram averaged acrossed the 50 oscillators
data = Array(sol)
spec = spectrogram(wrapTo2Pi(data[1, :]), div(150001, 200); fs=1000)

freq = spec.freq
time = spec.time
hmmpower = spec.power

for i = 2:50
    spec = spectrogram(wrapTo2Pi(data[i, :]), div(150001, 200); fs=1000)
    hmmpower .+= spec.power
end

using Plots
# This should reproduce Fig. 2A. Instead, what I'm seeing is oscillating around 
# the mean value (~20 Hz) as in the paper for the first 50s, then a jump up to a mean
# around the stimulation frequency (130Hz) but no emergent coherence at the 300Hz range.
heatmap(time, freq, pow2db.(hmmpower))

# To check if the coupling matches the values in Fig. 2D, compute equation 5 from the three
# neurotransmitter pools.
kₜ = zeros(length(sol))
for i = 1:length(sol)
    kₜ[i] = max(data[N+1, i], data[N+2, i]*0.5, data[N+3, i]*0.3)
end

# This should reproduce the blue line in Fig. 2D. The shape is roughly accurate, but the 
# scale is entirely off. I think this is because the authors dropped a factor in connecting
# the DBS stimulator to the neurotransmitter pools. Running just that simulation shows that
# they're very far off the actual values while preserving the shape of the dynamics.
plot(sol.t, kₜ .* 11976.0)

# Transmitter pool only simulation
# This *should* reproduce Figure 1B, but it doesn't. That's why I'm convinced there's a term
# missing in equation 4. The shape is correct, but the scale is off.

# Setup neurotransmitter pools
@named RRP = SermonNPool(τ_pool=1.0, p_pool=0.3)
@named RP  = SermonNPool(τ_pool=10.0, p_pool=0.02)
@named RtP = SermonNPool(τ_pool=50.0, p_pool=0.00035)

# Next, create the DBS stimulator block. This block is a simple pulse with a given width
# and period. 
@named DBS = SermonDBS(start_time=50.0, pulse_width=0.0001)

p = @parameters M_RRP=1.0 M_RP=0.5 M_RtP = 0.3 k_μ = 11976.0 

g = MetaDiGraph()
add_blox!.(Ref(g), vcat(DBS, RRP, RP, RtP))
add_edge!(g, 1, 2, Dict(:weight => 1.0))
add_edge!(g, 1, 3, Dict(:weight => 1.0))
add_edge!(g, 1, 4, Dict(:weight => 1.0))

@named sys = system_from_graph(g, p)
sys = structural_simplify(sys)

prob = ODEProblem(sys, [], (0.0, 150.0))
sol = solve(prob, Euler(), dt=0.0001, saveat=0.001)

# This should reproduce Fig. 1B. It doesn't :()
plot(sol)
