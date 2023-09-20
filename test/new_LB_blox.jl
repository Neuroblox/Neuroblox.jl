using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, LinearAlgebra, Random, DataFrames, Test


function LinearConnectionsLB(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
		norm_factor = sum(adj_matrix[:, region_num])
    	push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num])/norm_factor)
    end
    return @named Circuit = ODESystem(eqs, systems = sys)
end

adj = [0 1 0 0 0;
       0 0 1 0 0;
       0 0 0 1 0;
       0 0 0 0 1;
       1 0 0 0 0]

blox = []

for i = 1:5
    lb = LarterBreakspearBlox(name=Symbol("LB$i"))
    push!(blox,lb)
end

sys = [b.odesystem for b in blox]
con = [s.connector for s in blox]
@named LB_circuit_lin = LinearConnectionsLB(sys=sys, adj_matrix=adj,connector=con)
mysys = structural_simplify(LB_circuit_lin)

uw = rand(Int(length(mysys.states)/3))
uv = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
uz = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
u0 = collect(Iterators.flatten(zip(uv,uz,uw)))

prob = ODEProblem(mysys,u0,(0.0,5e2),[])
sol = solve(prob,AutoVern7(Rodas4()),saveat=0.1)
dsol = DataFrame(sol)

blox2 = []

for i = 1:5
    lb = LarterBreakspear(name=Symbol("LB$i"))
    push!(blox2,lb)
end

adj_temp = adj

g = MetaDiGraph()
add_blox!.(Ref(g), blox2)
create_adjacency_edges!(g, adj_temp)
@named mysys2 = system_from_graph(g)
mysys2 = structural_simplify(mysys2)
prob2 = ODEProblem(mysys2,u0,(0.0,5e2),[])
sol2 = solve(prob2,AutoVern7(Rodas4()),saveat=0.1)
dsol2 = DataFrame(sol2)

@test isapprox(dsol2[!, "LB2₊V(t)"][100:end], dsol[!, "LB2₊V(t)"][100:end], rtol=1e-8)