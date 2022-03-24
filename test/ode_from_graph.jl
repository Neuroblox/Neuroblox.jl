using Neuroblox, Test, SparseArrays, Graphs, MetaGraphs

@named GPe = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)

# Connect Regions through Adjacency Matrix
sys = [GPe, STN]
@parameters g_GPe_STN=1.0 g_STN_GPe=1.0
adj_matrix = [1.0 g_STN_GPe;
            g_STN_GPe*g_GPe_STN 1.0]

#create equivalent graph
g = LinearNeuroGraph(MetaDiGraph())
add_blox!(g,GPe)
add_blox!(g,STN)
add_edge!(g,1,1,:weight,1.0)
add_edge!(g,1,2,:weight,g_STN_GPe)
add_edge!(g,2,1,:weight,g_STN_GPe*g_GPe_STN)
add_edge!(g,2,2,:weight,1.0)
a = AdjMatrixfromLinearNeuroGraph(g)

# I am not sure how to test that the two adjacency matrices are equal
@test isequal(a,adj_matrix)

connector = [s.x for s in sys]
@named two_regions = LinearConnections(sys=sys,adj_matrix=adj_matrix, connector=connector)
@named two_regions_gr = ODEfromGraph(g=g)

@test typeof(two_regions) == ODESystem
@test typeof(two_regions_gr) == ODESystem
