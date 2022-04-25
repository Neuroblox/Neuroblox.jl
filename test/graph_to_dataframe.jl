using Neuroblox, Graphs, MetaGraphs, DataFrames, Test

"""
harmonic oscillator neural mass blox
"""
@named GPe = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)

# Connect Regions through Adjacency Matrix
@parameters g_GPe_STN=1.0 g_STN_GPe=1.0

#create graph
g = LinearNeuroGraph(MetaDiGraph())
add_blox!(g,GPe)
add_blox!(g,STN)
add_edge!(g,1,1,:weight,1.0)
add_edge!(g,1,2,:weight,g_STN_GPe)
add_edge!(g,2,1,:weight,g_STN_GPe*g_GPe_STN)
add_edge!(g,2,2,:weight,1.0)

@named two_regions_gr = ODEfromGraph(g=g)

sim_dur = 5.0 # Simulate for 10 Seconds

# returns dataframe with time series for 4 outputs
sol = simulate(structural_simplify(two_regions_gr), [], (0.0, sim_dur), [])
@test typeof(sol) == DataFrame

"""
jansen rit neural mass blox
"""

@named Str = jansen_ritSC(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_ritSC(τ=0.04, H=20, λ=400, r=0.1)
@named STN = jansen_ritSC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)

# Connect Regions through Adjacency Matrix
@parameters C_Cor=3 C_BG_Th=3 C_Cor_BG_Th=9.75 C_BG_Th_Cor=9.75

# Create Graph
g = LinearNeuroGraph(MetaDiGraph())
add_blox!(g,Str)
add_blox!(g,GPe)
add_blox!(g,STN)
add_blox!(g,GPi)
add_edge!(g,1,1,:weight,0.0)
add_edge!(g,1,2,:weight,0.0)
add_edge!(g,1,3,:weight,0.0)
add_edge!(g,1,4,:weight,0.0)
add_edge!(g,2,1,:weight,-0.5*C_BG_Th)
add_edge!(g,2,2,:weight,-0.5*C_BG_Th)
add_edge!(g,2,3,:weight,C_BG_Th)
add_edge!(g,2,4,:weight,0.0)
add_edge!(g,3,1,:weight,0.0)
add_edge!(g,3,2,:weight,-0.5*C_BG_Th)
add_edge!(g,3,3,:weight,0.0)
add_edge!(g,3,4,:weight,0.0)
add_edge!(g,4,1,:weight,0.0)
add_edge!(g,4,2,:weight,-0.5*C_BG_Th)
add_edge!(g,4,3,:weight, C_BG_Th)
add_edge!(g,4,4,:weight,0.0)

@named four_regions_gr = ODEfromGraph(g=g)

sim_dur = 5.0 # Simulate for 5 Seconds

# returns dataframe with time series for 2*4 outputs
sol = simulate(structural_simplify(four_regions_gr), [], (0.0, sim_dur), [])
@test typeof(sol) == DataFrame
