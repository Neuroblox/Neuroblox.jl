using Neuroblox, Graphs, MetaGraphs

@named Str = jansen_ritC(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
@named GPi = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1.0)

# Connect Regions through Adjacency Matrix
@parameters C_Cor=2.0
# Create Graph
g = LinearNeuroGraph(MetaDiGraph())
add_blox!(g,Str)
add_blox!(g,GPe)
add_blox!(g,STN)
add_blox!(g,GPi)
add_edge!(g,3,1,:weight,0.5*C_Cor)
add_edge!(g,4,1,:weight,0.5*C_Cor)
add_edge!(g,1,2,:weight,0.1*C_Cor)
add_edge!(g,3,4,:weight,0.1*C_Cor)
add_edge!(g,1,3,:weight,0.1*C_Cor)
add_edge!(g,2,4,:weight,0.1*C_Cor)

@named four_regions_gr = ODEfromGraph(g=g)

sim_dur = 10.0 # Simulate for 10 Seconds

# returns dataframe with time series for 2*4 outputs
sol = simulate(structural_simplify(four_regions_gr), [], (0.0, sim_dur), [])