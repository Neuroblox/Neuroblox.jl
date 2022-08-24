### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 44e6a44a-d577-11ec-384e-23dbb7b36131
begin
	import Pkg
	Pkg.add("CSV")
	Pkg.add("DataFrames")
	Pkg.add("MAT")
	Pkg.develop(path=joinpath(@__DIR__, "..", "..","..", "Neuroblox.jl"))
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, CSV, DataFrames,LinearAlgebra, MAT, Random
end

# ╔═╡ 9b4e0bc6-4af6-4311-be1b-5948ec818b29
connectivity = CSV.read(joinpath(@__DIR__,"connectivity_96/weights.txt"), DataFrame;
				types=Float64,
				header=false,
				delim=' ',
				ignorerepeated = true)

# ╔═╡ 34b168a9-529f-4b9f-a616-713f6a0d644b
connLB = matread(joinpath(@__DIR__, "DATA","mean_structural_connectivity.mat"))

# ╔═╡ a22cd490-b682-482e-9601-3ac82f49770a
clb = connLB["mean_structural_connectivity"]

# ╔═╡ f43cde71-80f3-462b-b792-0be92f9f28bc
0.140202/sum(clb[:,1])

# ╔═╡ ef5dde28-2525-452e-9428-08f2708c36f5
c = Matrix{Float64}(connectivity)

# ╔═╡ 706e0021-e86d-4e96-8b79-87277c628afd
begin
	blox = []
	for i = 1:78
	    lb = LauterBreakspearBlox(name=Symbol("LB$i"))
	    push!(blox,lb)
	end
end

# ╔═╡ 1abbb345-e14c-4501-9717-c09b25d73e91
sys = [b.odesystem for b in blox]

# ╔═╡ 4e617a58-6887-4782-941a-b2e387272d5b
function LinearConnectionsWC(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
		norm_factor = sum(adj_matrix[:, region_num])
		# @show num_conn
    	push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num])/norm_factor)
    end
    return @named Circuit = ODESystem(eqs, systems = sys)
end

# ╔═╡ abac20d3-bdcb-4ed6-b807-4278a72bc0d3
connect = [s.connector for s in blox]

# ╔═╡ 99dba142-6ee9-4ffa-b9dc-a377c0de9f2a
@named LB_Circuit_lin = LinearConnectionsWC(sys=sys, adj_matrix=clb, connector=connect)

# ╔═╡ eb905a35-b314-4aa4-8f31-bdf382d8f528
equations(LB_Circuit_lin)[1]

# ╔═╡ 69920cca-0cd3-4761-92f5-8dcca67500dc
mysys = structural_simplify(LB_Circuit_lin)

# ╔═╡ d8f367ad-52dd-4cc2-9c65-0f84cd22d62e
mysys.states

# ╔═╡ d593209e-2fc8-40c0-883b-935b5512c779
mysys.ps

# ╔═╡ 41587582-f793-4198-af39-7cd0637d57cc
md"""
coup 
$(@bind coup html"<input type=range value=0.5 min=0 max=1 step=0.01>")
delta
$(@bind delta html"<input type=range value=0.1 min=0 max=1 step=0.01>")
"""

# ╔═╡ 649e0326-64d8-436e-818d-2705d16a74a0
begin
	uw = 0.4*rand(Int(length(mysys.states)/3)) .+ 0.11
	uv = 0.9*rand(Int(length(mysys.states)/3)) .- 0.6
	uz = 0.9*rand(Int(length(mysys.states)/3)) .- 0.9
	u0 = collect(Iterators.flatten(zip(uv,uz,uw)))
end

# ╔═╡ 9778982a-e706-457f-8723-703d6fefbdb0
prob = ODEProblem(mysys,u0,(0.0,100.0),[])

# ╔═╡ b201b708-908a-4321-8c06-da3b442af069
begin
	p_new = prob.p
	for t_index in 1:78
		p_new[(t_index-1)*2+1] = coup
		p_new[(t_index-1)*2+2] = delta
	end
	prob2 = remake(prob;p=p_new)
	sol = solve(prob2,Rodas4(),saveat=0.1)
end

# ╔═╡ 4545b94d-fd25-4a3d-ba30-faeff8c89b8c
plot(sol,label=false,xlim=(0,100),ylim=(-1,3))

# ╔═╡ 1357d862-557b-4460-8c20-61c7ec9ef872
size(sol)

# ╔═╡ Cell order:
# ╠═44e6a44a-d577-11ec-384e-23dbb7b36131
# ╠═9b4e0bc6-4af6-4311-be1b-5948ec818b29
# ╠═34b168a9-529f-4b9f-a616-713f6a0d644b
# ╠═a22cd490-b682-482e-9601-3ac82f49770a
# ╠═f43cde71-80f3-462b-b792-0be92f9f28bc
# ╠═ef5dde28-2525-452e-9428-08f2708c36f5
# ╠═4e617a58-6887-4782-941a-b2e387272d5b
# ╠═706e0021-e86d-4e96-8b79-87277c628afd
# ╠═1abbb345-e14c-4501-9717-c09b25d73e91
# ╠═abac20d3-bdcb-4ed6-b807-4278a72bc0d3
# ╠═99dba142-6ee9-4ffa-b9dc-a377c0de9f2a
# ╠═eb905a35-b314-4aa4-8f31-bdf382d8f528
# ╠═69920cca-0cd3-4761-92f5-8dcca67500dc
# ╠═d8f367ad-52dd-4cc2-9c65-0f84cd22d62e
# ╠═d593209e-2fc8-40c0-883b-935b5512c779
# ╠═b201b708-908a-4321-8c06-da3b442af069
# ╠═41587582-f793-4198-af39-7cd0637d57cc
# ╠═649e0326-64d8-436e-818d-2705d16a74a0
# ╠═9778982a-e706-457f-8723-703d6fefbdb0
# ╠═4545b94d-fd25-4a3d-ba30-faeff8c89b8c
# ╠═1357d862-557b-4460-8c20-61c7ec9ef872
