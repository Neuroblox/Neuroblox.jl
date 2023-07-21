using Neuroblox, Test, SparseArrays, Graphs, MetaGraphs, CSV, DataFrames

adj_filename = "/Users/achesebro/Downloads/rois.csv" #change this to be selected from GUI
temp = splitext(adj_filename)
roi_filename = temp[1]*"_names.csv"
adj = Matrix(DataFrame(CSV.File(adj_filename, header=false)))
roi = DataFrame(CSV.File(roi_filename, header=false))
bloxType = "Neuroblox.JansenRitSCBlox" #change this to be selected from GUI; only JansenRit or LarterBreakspear for this draft -> HARD CODED TO LarterBreakspearBlox for now!

blox = []
for i in 1:size(adj,1)
	lb = LarterBreakspearBlox(name=Symbol(roi[i,1]))
	push!(blox,lb)
end

sys = [b.odesystem for b in blox]
con = [s.connector for s in blox]

# Not needed now, but will be once parameters can vary across regions 
@parameters T_Ca=-0.01 δ_Ca=0.15 g_Ca=1.0 V_Ca=1.0 T_K=0.0 δ_K=0.3 g_K=2.0 V_K=-0.7 T_Na=0.3 δ_Na=0.15 g_Na=6.7 V_Na=0.53 V_L=-0.5 g_L=0.5 V_T=0.0 Z_T=0.0 δ_VZ=0.61 Q_Vmax=1.0 Q_Zmax=1.0 IS=0.3 a_ee=0.36 a_ei=2.0 a_ie=2.0 a_ne=1.0 a_ni=0.4 b=0.1 τ_K=1.0 ϕ=0.7 r_NMDA=0.25 C=0.35

#create equivalent graph
g = MetaDiGraph()
for i in 1:size(adj,1)
    add_blox!(g,LarterBreakspearBlox(name=Symbol(roi[i,1])))
end

for i in 1:size(adj,1)
    for j in 1:size(adj,2)
        if adj[i,j] != 0
            add_edge!(g,i,j,:weight,adj[i,j])
        end
    end
end
a = adjmatrixfromdigraph(g)

# I am not sure how to test that the two adjacency matrices are equal
@test isequal(a,adj)

@named LB_circuit_lin = LinearConnections(sys=sys, adj_matrix=adj,connector=con)
@named LB_circuit_graph = ODEfromGraph(g)

@test typeof(LB_circuit_lin) == ODESystem
@test typeof(LB_circuit_graph) == ODESystem
@test equations(LB_circuit_lin) == equations(LB_circuit_graph)
