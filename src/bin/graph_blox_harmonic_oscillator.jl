using Neuroblox, Graphs, MetaGraphs

@named GPe = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1.0)

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

sim_dur = 10.0 # Simulate for 10 Seconds

# returns dataframe with time series for 4 outputs
sol = simulate(structural_simplify(two_regions_gr), [], (0.0, sim_dur), [])
