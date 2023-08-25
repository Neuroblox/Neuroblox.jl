using Neuroblox, Test, SparseArrays, Graphs, MetaGraphs

@named GPe = HarmonicOscillatorBlox()
@named STN = HarmonicOscillatorBlox()
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

# parameter sharing test
@parameters λ=400.0
@named jr1 = JansenRitSCBlox(τ=0.014, H=20.0, λ=λ, r=0.1)
@named jr2 = JansenRitSCBlox(τ=0.014, H=20.0, λ=λ, r=0.1)
jg = MetaDiGraph()
add_blox!(jg,jr1)
add_blox!(jg,jr2)
add_edge!(jg,1,2,:weight,1.0)
add_edge!(jg,2,1,:weight,2.0)
add_edge!(jg,1,1,:weight,-1.0)
add_edge!(jg,2,2,:weight,-2.0)
@named jg_graph = ODEfromGraph(jg)
jg_graph_s = structural_simplify(jg_graph)
@test length(parameters(jg_graph_s))==7