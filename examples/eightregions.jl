### A Pluto.jl notebook ###
# v0.19.0

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
	Pkg.add(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs
end

# ╔═╡ 66460e85-1544-406b-9c79-773ab174a5cb
begin
	# Create Regions
	@named Str = jansen_ritSC(τ=0.0022, H=20, λ=300, r=0.3)
	@named GPe = jansen_ritSC(τ=0.04, H=20, λ=400, r=0.1)
	@named STN = jansen_ritSC(τ=0.01, H=20, λ=500, r=0.1)
	@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)
	@named Th  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
	@named EI  = jansen_ritC(τ=0.01, H=20, λ=5, r=5)
	@named PY  = jansen_ritC(τ=0.001, H=20, λ=5, r=0.15)
	@named II  = jansen_ritC(τ=2.0, H=60, λ=5, r=5)
end

# ╔═╡ 9aa9ae2b-b8a0-463d-8e9e-b3339b25a99d
@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

# ╔═╡ fef8036a-f073-4473-9549-2b69ec644e8a
begin
	g = LinearNeuroGraph(MetaDiGraph())
	add_blox!(g,Str)
	add_blox!(g,GPe)
	add_blox!(g,STN)
	add_blox!(g,GPi)
	add_blox!(g,Th)
	add_blox!(g,EI)
	add_blox!(g,PY)
	add_blox!(g,II)

	add_edge!(g,2,1,:weight, -0.5*C_BG_Th)
	add_edge!(g,2,2,:weight, -0.5*C_BG_Th)
	add_edge!(g,2,3,:weight, C_BG_Th)
	
	add_edge!(g,3,2,:weight, -0.5*C_BG_Th)
	add_edge!(g,3,7,:weight, C_Cor_BG_Th)
	
	add_edge!(g,4,2,:weight, -0.5*C_BG_Th)
	add_edge!(g,4,3,:weight, C_BG_Th)
	
	add_edge!(g,5,4,:weight, -0.5*C_BG_Th)
	
	add_edge!(g,6,5,:weight, C_BG_Th_Cor)
	add_edge!(g,6,7,:weight, 6*C_Cor)
	
	add_edge!(g,7,6,:weight, 4.8*C_Cor)
	add_edge!(g,7,8,:weight, -1.5*C_Cor)

	add_edge!(g,8,7,:weight, 1.5*C_Cor)
	add_edge!(g,8,8,:weight, -3.3*C_Cor)
end

# ╔═╡ 5fc63975-3a15-4430-a7f1-4e5db64c04a1
AdjMatrixfromLinearNeuroGraph(g)

# ╔═╡ 785d8e9a-9ec0-4a5d-92a4-1d1d18d4ffc0
adj_matrix_lin = [0 0 0 0 0 0 0 0;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
            0 0 0 0 0 0 1.5*C_Cor -3.3*C_Cor]

# ╔═╡ a5faad0e-8db0-4bef-b344-355b5cc8ef0f
begin
	blox = [ get_prop(g.graph,v,:blox) for v in 1:nv(g.graph)]
    sys = [s.odesystem for s in blox]
    connector = [s.connector for s in blox]
end

# ╔═╡ 6d4463a3-a13a-4adb-950f-0e2115d92b8f
blox

# ╔═╡ 3ae6df4c-25cb-46b9-940c-4cd5a811689e
begin
	adj = adj_matrix_lin .* connector
    eqs = []
    for region_num in 1:length(sys)
            push!(eqs, sys[region_num].jcn ~ sum(adj[:,region_num]))
    end
	@named eightregions_odesys = ODESystem(eqs, systems = sys)
end

# ╔═╡ 33002a2b-f8a9-4728-8288-2f92d3b89948
@named eight_regions_gr = ODEfromGraph(g=g)

# ╔═╡ c4b4aa78-0324-4ea8-9903-efe87f6074e8
eight_regions_s = structural_simplify(eightregions_odesys)

# ╔═╡ 1a48d894-f43b-4559-8844-50b6e1989bda
sim_dur = 10.0 # Simulate for 10 Seconds

# ╔═╡ 906a3f36-613c-465c-b7d1-6caa245cfe86
prob = ODAEProblem(eight_regions_s, [], (0,sim_dur), [])

# ╔═╡ dd455500-d1bf-443d-b589-d400b6844874
prob.p

# ╔═╡ 7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
eight_regions_s.ps

# ╔═╡ 0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
eight_regions_s.states

# ╔═╡ b94c0e24-1793-4fe0-b84d-974d0c27113c
md"""
BG-Th
$(@bind bgth html"<input type=range min=0 max=10000>")
Cor BG-Th
$(@bind corbgth html"<input type=range min=0 max=10000>")
BG-Th Cor
$(@bind bgthcor html"<input type=range min=0 max=10000>")

Cor
$(@bind cor html"<input type=range min=0 max=10000>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	p_new = prob.p
	p_new[1] = (bgth)/100
	p_new[2] = (corbgth-5000)/1000
	p_new[3] = (cor)/100
	p_new[4] = (bgthcor-5000)/1000
	prob_new = remake(prob; p=p_new, u0=ones(16)*0.1)
	sol = solve(prob_new, Tsit5())
end

# ╔═╡ c81dfd67-d338-43af-82b4-a83671c3148d
((bgth)/100,(corbgth-5000)/1000,(bgthcor-5000)/1000,(cor)/100)

# ╔═╡ 8accf027-0261-42e5-ac11-c066cfb57c43
begin
	l = @layout [a; b; c]
	p1 = plot(sol.t,sol[11,:],label="EI")
	p2 = plot(sol.t,sol[13,:],label="PY")
	p3 = plot(sol.t,sol[15,:],label="II")
	plot(p1, p2, p3, layout = l)
end

# ╔═╡ bcb92a18-166c-46a7-aace-ccca97a825e4
begin
	l2 = @layout [a; b; c; d; e]
	p4 = plot(sol.t,sol[1,:],label="Str")
	p5 = plot(sol.t,sol[3,:],label="GPe")
	p6 = plot(sol.t,sol[5,:],label="STN")
	p7 = plot(sol.t,sol[7,:],label="GPi")
	p8 = plot(sol.t,sol[9,:],label="Th")
	plot(p4, p5, p6, p7, p8, layout = l2)
end

# ╔═╡ Cell order:
# ╠═771d1460-c48a-11ec-10d4-c7c5dd2a9984
# ╠═66460e85-1544-406b-9c79-773ab174a5cb
# ╠═9aa9ae2b-b8a0-463d-8e9e-b3339b25a99d
# ╠═fef8036a-f073-4473-9549-2b69ec644e8a
# ╠═5fc63975-3a15-4430-a7f1-4e5db64c04a1
# ╠═785d8e9a-9ec0-4a5d-92a4-1d1d18d4ffc0
# ╠═a5faad0e-8db0-4bef-b344-355b5cc8ef0f
# ╠═6d4463a3-a13a-4adb-950f-0e2115d92b8f
# ╠═3ae6df4c-25cb-46b9-940c-4cd5a811689e
# ╠═33002a2b-f8a9-4728-8288-2f92d3b89948
# ╠═c4b4aa78-0324-4ea8-9903-efe87f6074e8
# ╠═1a48d894-f43b-4559-8844-50b6e1989bda
# ╠═906a3f36-613c-465c-b7d1-6caa245cfe86
# ╠═dd455500-d1bf-443d-b589-d400b6844874
# ╠═7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
# ╠═0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
# ╠═32d1b83e-acfa-4351-9bcf-bc4367781186
# ╟─b94c0e24-1793-4fe0-b84d-974d0c27113c
# ╠═c81dfd67-d338-43af-82b4-a83671c3148d
# ╠═8accf027-0261-42e5-ac11-c066cfb57c43
# ╠═bcb92a18-166c-46a7-aace-ccca97a825e4
