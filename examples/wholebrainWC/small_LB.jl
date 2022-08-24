using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, CSV, DataFrames,LinearAlgebra, MAT, Random, Plots

clb = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]

function LinearConnectionsLB(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
		norm_factor = sum(adj_matrix[:, region_num])
		# @show num_conn
    	push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num])/norm_factor)
    end
    return @named Circuit = ODESystem(eqs, systems = sys)
end

blox = []
for i = 1:3
	lb = LauterBreakspearBlox(name=Symbol("LB$i"))
	push!(blox,lb)
end

sys = [b.odesystem for b in blox]
con = [s.connector for s in blox]
@named LB_circuit_lin = LinearConnectionsLB(sys=sys, adj_matrix=clb,connector=con)
mysys = structural_simplify(LB_circuit_lin)

mysys.states
mysys.ps

uw = rand(Int(length(mysys.states)/3))
uv = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
uz = 0.1*rand(Int(length(mysys.states)/3)) .- 0.05
u0 = collect(Iterators.flatten(zip(uv,uz,uw)))

prob = ODEProblem(mysys,u0,(0.0,6e5),[])

p_new = prob.p
for t_index in 1:3
	p_new[(t_index-1)*2+1] = 0.35
	p_new[(t_index-1)*2+2] = 0.65
end
prob2 = remake(prob;p=p_new)
sol2 = solve(prob2,Rodas5(),saveat=0.1)

plot(sol2,xlim=(0,600),ylim=(-1,1))
savefig("LBsmall.png")





