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

# ╔═╡ 771d1460-c48a-11ec-10d4-c7c5dd2a9984
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
	Pkg.develop(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, Printf
end

# ╔═╡ 66460e85-1544-406b-9c79-773ab174a5cb
begin
	# Create Regions
	@named WC1 = wilson_cowan()
	@named WC2 = wilson_cowan()
end

# ╔═╡ 9f3d6efd-a884-449c-b4b8-42b0b435e245
begin
	blox = [WC1,WC2]
	sys = [b.system for b in blox]
	connect = [b.connector for b in blox]
end

# ╔═╡ c4b4aa78-0324-4ea8-9903-efe87f6074e8
@parameters c[1:2,1:2]=[1.0 1.0;1.0 1.0]

# ╔═╡ c67c098c-0829-403f-98b3-adb71b387a12
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

# ╔═╡ 906a3f36-613c-465c-b7d1-6caa245cfe86
@named WC_sys = LinearConnectionsWC(sys=sys, adj_matrix = c, connector=connect)

# ╔═╡ dd455500-d1bf-443d-b589-d400b6844874
WC_sys_simpl = structural_simplify(WC_sys)

# ╔═╡ 7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
WC_sys_simpl.states

# ╔═╡ 0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
WC_sys_simpl.ps

# ╔═╡ a1c1b45f-8692-4456-ab20-d72b3e44fc0d
prob = ODEProblem(WC_sys_simpl,0.5*ones(length(WC_sys_simpl.states)),(0.0,100.0),[])

# ╔═╡ 3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
md"""
c11
$(@bind c11 html"<input type=range value=3.3 min=0 max=50 step=0.1>")
c12
$(@bind c12 html"<input type=range value=6.7 min=0 max=50 step=0.1>")
"""

# ╔═╡ 5dc52090-4f13-49d5-b39e-383520a067df
md"""
c21 
$(@bind c21 html"<input type=range value=2.3 min=0 max=50 step=0.1>")
c22
$(@bind c22 html"<input type=range value=0.0 min=0 max=50 step=0.1>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	# set the parameters for the simulation using the sliders
	p_new = prob.p
	p_new[1] = c11
	p_new[2] = c21
	p_new[3] = c12
	p_new[4] = c22
	prob_new = remake(prob; p=p_new)
	sol = solve(prob_new, Rodas4())
end

# ╔═╡ 71efa2e3-5719-44ec-b383-68e4304c4c1b
c11,c12,c21,c22

# ╔═╡ 8accf027-0261-42e5-ac11-c066cfb57c43
begin
	l = @layout [a; b]
	p1 = plot(sol.t,sol[1,:],label="E")
	p2 = plot(sol.t,sol[2,:],label="I")
	plot(p1, p2, layout = l)
end

# ╔═╡ Cell order:
# ╠═771d1460-c48a-11ec-10d4-c7c5dd2a9984
# ╠═66460e85-1544-406b-9c79-773ab174a5cb
# ╠═9f3d6efd-a884-449c-b4b8-42b0b435e245
# ╠═c4b4aa78-0324-4ea8-9903-efe87f6074e8
# ╠═c67c098c-0829-403f-98b3-adb71b387a12
# ╠═906a3f36-613c-465c-b7d1-6caa245cfe86
# ╠═dd455500-d1bf-443d-b589-d400b6844874
# ╠═7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
# ╠═0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
# ╠═a1c1b45f-8692-4456-ab20-d72b3e44fc0d
# ╠═32d1b83e-acfa-4351-9bcf-bc4367781186
# ╟─3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
# ╠═5dc52090-4f13-49d5-b39e-383520a067df
# ╠═71efa2e3-5719-44ec-b383-68e4304c4c1b
# ╠═8accf027-0261-42e5-ac11-c066cfb57c43
