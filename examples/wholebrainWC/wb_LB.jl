using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, CSV, DataFrames,LinearAlgebra, MAT, Random, Plots

connLB = matread(joinpath(@__DIR__, "DATA","mean_structural_connectivity.mat"))
clb = connLB["mean_structural_connectivity"]

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
for i = 1:78
	lb = LauterBreakspearBlox(name=Symbol("LB$i"))
	push!(blox,lb)
end

sys = [b.odesystem for b in blox]
connect = [s.connector for s in blox]
@named LB_circuit_lin = LinearConnectionsLB(sys=sys, adj_matrix=clb,connector=connect)
mysys = structural_simplify(LB_circuit_lin)

mysys.states
mysys.ps

uw = 0.4*rand(Int(length(mysys.states)/3)) .+ 0.11
uv = 0.9*rand(Int(length(mysys.states)/3)) .- 0.6
uz = 0.9*rand(Int(length(mysys.states)/3)) .- 0.9
u0 = collect(Iterators.flatten(zip(uv,uz,uw)))

prob = ODEProblem(mysys,u0,(0.0,100.0),[])

p_new = prob.p
for t_index in 1:78
	p_new[(t_index-1)*2+1] = 0.35
	p_new[(t_index-1)*2+2] = 0.65
end
prob2 = remake(prob;p=p_new)
sol2 = solve(prob2,Rodas4(),saveat=0.1)

plot(sol,label=false,xlim=(0,100),ylim=(-1,3))






