using Neuroblox, Graphs, MetaGraphs

@named Str = jansen_rit(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_rit(τ=0.04, H=20, λ=400, r=0.1)
@named STN = jansen_rit(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = jansen_rit(τ=0.014, H=20, λ=400, r=0.1)

# Connect Regions through Adjacency Matrix
sys = [Str, GPe, STN, GPi]
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

sim_dur = 10.0 # Simulate for 10 Seconds

# returns dataframe with time series for 2*4 outputs
sol = simulate(structural_simplify(four_regions_gr), [], (0.0, sim_dur), [])