### A Pluto.jl notebook ###
# v0.19.4

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
    # activate the shared project environment
    Pkg.activate(@__DIR__)
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("OrdinaryDiffEq")
	Pkg.add("MetaGraphs")
	Pkg.add("Graphs")
	Pkg.add("Plots")
	Pkg.add("CSV")
	Pkg.add("DataFrames")
	Pfg.add("LinearAlgebra")
	Pkg.develop(path=joinpath(@__DIR__, "..", "..","..", "Neuroblox.jl"))
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, CSV, DataFrames,LinearAlgebra
end

# ╔═╡ 9b4e0bc6-4af6-4311-be1b-5948ec818b29
connectivity = CSV.read(joinpath(@__DIR__,"connectivity_96/weights.txt"), DataFrame;
				types=Float64,
				header=false,
				delim=' ',
				ignorerepeated = true)

# ╔═╡ ef5dde28-2525-452e-9428-08f2708c36f5
c = Matrix{Float64}(connectivity)

# ╔═╡ b208f23a-69c3-4de0-b0c8-52c36808952e
begin
	c_nt = copy(c)
	c_nt[diagind(c_nt)] .= 0.0
end

# ╔═╡ 512ea317-ce6c-4791-81f6-0fbde4b3b646
c_nt

# ╔═╡ 706e0021-e86d-4e96-8b79-87277c628afd
begin
	blox = []
	for i = 1:96
	    wc = wilson_cowan(name=Symbol("WC$i"),
							τ_E=0.5,
							τ_I=1.0,
							a_E=1.0,
	                        a_I=1.0,
	                        c_EE=4.0,
	                        c_IE=30.0,
	                        c_EI=30.0,
	                        c_II=10.0,
	                        θ_E=0.0,
	                        θ_I=0.0,
	                        η=0.4)
	    push!(blox,wc)
	end
end

# ╔═╡ 1abbb345-e14c-4501-9717-c09b25d73e91
sys = [b.odesystem for b in blox]

# ╔═╡ 4e617a58-6887-4782-941a-b2e387272d5b
function LinearConnectionsWC(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
		#num_conn = 1.0*count(!iszero,adj_matrix[:, region_num])
		# @show num_conn
    	push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num]))
		push!(eqs, sys[region_num].P ~ 0.0)
    end
    return @named Circuit = ODESystem(eqs, systems = sys)
end

# ╔═╡ abac20d3-bdcb-4ed6-b807-4278a72bc0d3
connect = [s.connector for s in blox]

# ╔═╡ 99dba142-6ee9-4ffa-b9dc-a377c0de9f2a
@named WC_Circuit_lin = LinearConnectionsWC(sys=sys, adj_matrix=c_nt, connector=connect)

# ╔═╡ 69920cca-0cd3-4761-92f5-8dcca67500dc
mysys = structural_simplify(WC_Circuit_lin)

# ╔═╡ d8f367ad-52dd-4cc2-9c65-0f84cd22d62e
length(mysys.states)

# ╔═╡ d593209e-2fc8-40c0-883b-935b5512c779
mysys.ps

# ╔═╡ 9778982a-e706-457f-8723-703d6fefbdb0
prob = ODEProblem(mysys,0.5*ones(length(mysys.states)),(0.0,100.0),[])

# ╔═╡ c8a72998-f76a-4bad-acd9-d16c1a997415
md"""
tau_E
$(@bind tauE html"<input type=range value=1.0 min=0.1 max=10.0 step=0.1>")
tau_I
$(@bind tauI html"<input type=range value=0.9 min=0.1 max=10.0 step=0.1>")
eta
$(@bind eta html"<input type=range value=0.4 min=0.0 max=30.0 step=0.1>")
"""

# ╔═╡ 9e0007f9-829b-4e70-b38a-ee9120daafdb
md"""
WC₊θ_E
$(@bind tE html"<input type=range value=16.7 min=0 max=50 step=0.1>")
WC₊θ_I
$(@bind tI html"<input type=range value=13.9 min=0 max=50 step=0.1>")
"""

# ╔═╡ 4feee4e3-e36c-448f-84de-88f61b586c91
md"""
WC₊c_EE, 
$(@bind cEE html"<input type=range value=26.8 min=0 max=50 step=0.1>")
WC₊c_IE
$(@bind cIE html"<input type=range value=26 min=0 max=50 step=0.1>")
"""

# ╔═╡ 41587582-f793-4198-af39-7cd0637d57cc
md"""
WC₊c_II, 
$(@bind cII html"<input type=range value=31.9 min=0 max=50 step=0.1>")
WC₊c_EI
$(@bind cEI html"<input type=range value=11.4 min=0 max=50 step=0.1>")
"""

# ╔═╡ b201b708-908a-4321-8c06-da3b442af069
begin
	p_new = prob.p
	for t_index in 1:96
		p_new[(t_index-1)*11+1] = tauE
		p_new[(t_index-1)*11+2] = tauI
		p_new[(t_index-1)*11+5] = cEE
		p_new[(t_index-1)*11+6] = cIE
		p_new[(t_index-1)*11+7] = cII
		p_new[(t_index-1)*11+8] = cEI
		p_new[(t_index-1)*11+9] = tE
		p_new[(t_index-1)*11+10] = tI
		p_new[(t_index-1)*11+11] = eta
	end
	prob2 = remake(prob;p=p_new)
	sol = solve(prob2,Rodas4(),saveat=0.1)
end

# ╔═╡ 469f3bd1-18f7-42f8-8972-3e9a4713bf66


# ╔═╡ 20c27485-3265-49b3-bb9e-4867b539a71e
tauE,tauI, tE,tI,cEE,cIE,cII,cEI,eta

# ╔═╡ 4545b94d-fd25-4a3d-ba30-faeff8c89b8c
plot(sol,label=false)

# ╔═╡ 1357d862-557b-4460-8c20-61c7ec9ef872
size(sol)

# ╔═╡ Cell order:
# ╠═44e6a44a-d577-11ec-384e-23dbb7b36131
# ╠═9b4e0bc6-4af6-4311-be1b-5948ec818b29
# ╠═ef5dde28-2525-452e-9428-08f2708c36f5
# ╠═b208f23a-69c3-4de0-b0c8-52c36808952e
# ╠═512ea317-ce6c-4791-81f6-0fbde4b3b646
# ╠═4e617a58-6887-4782-941a-b2e387272d5b
# ╠═706e0021-e86d-4e96-8b79-87277c628afd
# ╠═1abbb345-e14c-4501-9717-c09b25d73e91
# ╠═abac20d3-bdcb-4ed6-b807-4278a72bc0d3
# ╠═99dba142-6ee9-4ffa-b9dc-a377c0de9f2a
# ╠═69920cca-0cd3-4761-92f5-8dcca67500dc
# ╠═d8f367ad-52dd-4cc2-9c65-0f84cd22d62e
# ╠═d593209e-2fc8-40c0-883b-935b5512c779
# ╠═9778982a-e706-457f-8723-703d6fefbdb0
# ╠═b201b708-908a-4321-8c06-da3b442af069
# ╟─c8a72998-f76a-4bad-acd9-d16c1a997415
# ╟─9e0007f9-829b-4e70-b38a-ee9120daafdb
# ╟─4feee4e3-e36c-448f-84de-88f61b586c91
# ╠═41587582-f793-4198-af39-7cd0637d57cc
# ╠═469f3bd1-18f7-42f8-8972-3e9a4713bf66
# ╠═20c27485-3265-49b3-bb9e-4867b539a71e
# ╠═4545b94d-fd25-4a3d-ba30-faeff8c89b8c
# ╠═1357d862-557b-4460-8c20-61c7ec9ef872
