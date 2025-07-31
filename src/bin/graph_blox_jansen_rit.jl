using Neuroblox, Graphs, MetaGraphs

# Create Regions
@named GPe       = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
@named STN       = jansen_ritC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi       = jansen_ritC(τ=0.014, H=20, λ=400, r=0.1)
@named Thalamus  = jansen_ritC(τ=0.002, H=10, λ=20, r=5)
@named PFC       = jansen_ritC(τ=0.001, H=20, λ=5, r=0.15)

# Connect Regions through Adjacency Matrix
@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5
# Create Graph
g = LinearNeuroGraph(MetaDiGraph())
add_blox!(g,GPe)
add_blox!(g,STN)
add_blox!(g,GPi)
add_blox!(g,Thalamus)
add_blox!(g,PFC)

add_edge!(g,1,1,:weight, -0.5*C_BG_Th)
add_edge!(g,1,2,:weight, C_BG_Th)
add_edge!(g,2,1,:weight, -0.5*C_BG_Th)
add_edge!(g,2,5,:weight, C_Cor_BG_Th)
add_edge!(g,3,1,:weight, -0.5*C_BG_Th)
add_edge!(g,3,2,:weight, C_BG_Th)
add_edge!(g,4,3,:weight, -0.5*C_BG_Th)
add_edge!(g,4,4,:weight, C_BG_Th_Cor)


@named five_regions_gr = ODEfromGraph(g=g)
sim_dur = 10.0 # Simulate for 10 Seconds

# returns dataframe with time series for 2*5 outputs
sol = simulate(structural_simplify(five_regions_gr), [], (0.0, sim_dur), [])