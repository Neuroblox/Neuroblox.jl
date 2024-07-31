using Neuroblox
using DifferentialEquations
using DataFrames
using Distributions
using Statistics
using LinearAlgebra
using Graphs
using MetaGraphs
using Random

# First, create the three neurotransmitter pools. As they have the same structure,
# we can use the same constructor with different parameters to create the systems.
@named RRP = SermonNPool(τ_pool=1.0, p_pool=0.3)
@named RP  = SermonNPool(τ_pool=10.0, p_pool=0.02)
@named RtP = SermonNPool(τ_pool=50.0, p_pool=0.00035)

# Next, create the DBS stimulator block. This block is a simple pulse with a given width
# and period. 
@named DBS = SermonDBS()

# Next, create a collection of 50 Kuramotor oscillators with noise.
# Set the number of oscillators
N = 50

# Define the natural distribution of oscillator frequencies
Ω = 249
σ = 26.317

ks_blocks = [KuramotoOscillator(name=Symbol("KO$i"), 
                                ω=rand(Normal(Ω, σ)),
                                ζ=5.920,
                                include_noise=true) for i in 1:N]

# Create a graph and add all the oscillators to it
g = MetaDiGraph()
add_blox!.(Ref(g), ks_blocks)

# Connect all oscillators to each other
for i in 1:N
    for j in 1:N
        add_edge!(g, i, j, Dict(:weight => 1.0, :sermon_rule => true))
    end
end

@named sys = system_from_graph(g)
@time sys = structural_simplify(sys)


sim_len = 150.0
@time prob = SDEProblem(sys, [], (0.0, sim_len))
@time sol = solve(prob, SRA1(), dt=0.0001, saveat=0.001)