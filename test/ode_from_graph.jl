using Neuroblox, Test, SparseArrays, Graphs, MetaGraphs

@named GPe = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = harmonic_oscillator(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)
@named BP = BandPassFilterBlox()

# Connect Regions through Adjacency Matrix
blox = [GPe, STN]
sys = [s.odesystem for s in blox]
connect = [s.connector for s in blox]

@parameters g_GPe_STN=1.0 g_STN_GPe=1.0
adj_matrix = [1.0 g_STN_GPe;
            g_STN_GPe*g_GPe_STN 1.0]

#create equivalent graph
g = MetaDiGraph()
add_blox!(g,GPe)
add_blox!(g,STN)
add_edge!(g,1,1,:weight,1.0)
add_edge!(g,1,2,:weight,g_STN_GPe)
add_edge!(g,2,1,:weight,g_STN_GPe*g_GPe_STN)
add_edge!(g,2,2,:weight,1.0)
a = adjmatrixfromdigraph(g)

#create equivalent graph with added Utilities Blox
ggb = MetaDiGraph()
add_blox!(ggb,GPe)
add_blox!(ggb,STN)
add_blox!(ggb,BP)
add_edge!(ggb,1,1,:weight,1.0)
add_edge!(ggb,1,2,:weight,g_STN_GPe)
add_edge!(ggb,2,1,:weight,g_STN_GPe*g_GPe_STN)
add_edge!(ggb,2,2,:weight,1.0)
add_edge!(ggb,2,3,:weight,1.0)

# I am not sure how to test that the two adjacency matrices are equal
@test isequal(a,adj_matrix)

@named two_regions = LinearConnections(sys=sys,adj_matrix=adj_matrix, connector=connect)
@named two_regions_gr = ODEfromGraph(g)
@named two_regions_grdb = ODEfromGraph(ggb)

@test typeof(two_regions) == ODESystem
@test typeof(two_regions_gr) == ODESystem
@test typeof(two_regions_grdb) == ODESystem
@test equations(two_regions) == equations(two_regions_gr)
@test equations(two_regions_gr) == equations(two_regions_grdb)

"""
test for HHNeuronExciBlox, HHNeuronInhibBlox and SynapticConnections
"""

nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_in=3, freq=4)
nn2 = HHNeuronExciBlox(name=Symbol("nrn2"), I_in=2, freq=6)
nn3 = HHNeuronInhibBlox(name=Symbol("nrn3"), I_in=2, freq=3)

assembly = [nn1, nn2, nn3]
adj = [0 1 0
       0 0 1
       0.2 0 0]
sys = [s.odesystem for s in assembly]
connect = [s.connector for s in assembly]       
@named neuron_net = SynapticConnections(sys=sys, adj_matrix=adj, connector=connect)

#create equivalent graph
gg = MetaDiGraph()
add_blox!(gg,nn1)
add_blox!(gg,nn2)
add_blox!(gg,nn3)
add_edge!(gg,1,2,:weight,1.0)
add_edge!(gg,2,3,:weight,1.0)
add_edge!(gg,3,1,:weight,.2)

@named neuron_net_graph = ODEfromGraph(gg)

# add your test to see whether you get the same ODESystem
@test typeof(neuron_net) == ODESystem
@test typeof(neuron_net_graph) == ODESystem
@test equations(neuron_net) == equations(neuron_net_graph)