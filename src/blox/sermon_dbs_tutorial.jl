using Neuroblox
using DifferentialEquations
using DataFrames
using Distributions
using Statistics
using LinearAlgebra
using Graphs
using MetaGraphs
using Random
using DSP
using ModelingToolkit: namespace_expr

# First, create the three neurotransmitter pools. As they have the same structure,
# we can use the same constructor with different parameters to create the systems.
@named RRP = SermonNPool(τ_pool=1.0, p_pool=0.3)
@named RP  = SermonNPool(τ_pool=10.0, p_pool=0.02)
@named RtP = SermonNPool(τ_pool=50.0, p_pool=0.00035)

# Next, create the DBS stimulator block. This block is a simple pulse with a given width
# and period. 
@named DBS = SermonDBS(start_time=50.0)

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
add_blox!.(Ref(g), vcat(ks_blocks, DBS, RRP, RP, RtP))

# p = @parameters M_RRP=1.0 M_RP=0.5 M_RtP = 0.3 k_μ = 11976.0 

# out_RRP = namespace_expr(RRP.output, get_namespaced_sys(RRP))
# out_RP = namespace_expr(RP.output, get_namespaced_sys(RP))
# out_RtP = namespace_expr(RtP.output, get_namespaced_sys(RtP))

# # Connect all oscillators to each other
# for i in 1:N
#     for j in 1:N
#         add_edge!(g, i, j, Dict(:weight => k_μ * max(M_RRP * out_RRP, M_RP * out_RP, M_RtP * out_RtP)/N, :sermon_rule => true))
#     end
# end

p = @parameters M_RRP=1.0 M_RP=0.5 M_RtP = 0.3 k_μ = 11976.0 

# Connect all oscillators to each other
for i in 1:N
    for j in 1:N
        add_edge!(g, i, j, Dict(:weight => k_μ/N, :sermon_rule => true, :extra_bloxs => [RRP, RP, RtP], :extra_params => p))
    end
end

for i in 1:N
    add_edge!(g, N+1, i, Dict(:weight => 72053.0))
end

add_edge!(g, N+1, N+2, Dict(:weight => 1.0))
add_edge!(g, N+1, N+3, Dict(:weight => 1.0))
add_edge!(g, N+1, N+4, Dict(:weight => 1.0))

@named sys = system_from_graph(g, p)
sys = structural_simplify(sys)

