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
# ╠═╡ show_logs = false
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
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs
end

# ╔═╡ 66460e85-1544-406b-9c79-773ab174a5cb
begin
	# Create Regions
	@named GPe = jansen_ritSC(τ=0.04, H=20, λ=400, r=0.1)
	@named STN = jansen_ritSC(τ=0.01, H=20, λ=500, r=0.1)
	@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)
	@named Th  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
	@named EI  = jansen_ritC(τ=0.01, H=20, λ=5, r=5)
	@named PY  = jansen_ritC(τ=0.001, H=20, λ=5, r=0.15)
	@named II  = jansen_ritC(τ=2.0, H=60, λ=5, r=5)
end

# ╔═╡ 9aa9ae2b-b8a0-463d-8e9e-b3339b25a99d
@parameters C_Cor=3 C_BG_Th=3 C_Cor_BG_Th=9.75 C_BG_Th_Cor=9.75

# ╔═╡ f45de893-522a-4c0d-b1f0-9093623208ee
begin
	g7 = LinearNeuroGraph(MetaDiGraph())
	add_blox!(g7,GPe)
	add_blox!(g7,STN)
	add_blox!(g7,GPi)
	add_blox!(g7,Th)
	add_blox!(g7,EI)
	add_blox!(g7,PY)
	add_blox!(g7,II)

	add_edge!(g7,1,1,:weight, -0.5*C_BG_Th)
	add_edge!(g7,2,1,:weight, C_BG_Th)
	
	add_edge!(g7,1,2,:weight, -0.5*C_BG_Th)
	add_edge!(g7,6,2,:weight, C_Cor_BG_Th)
	
	add_edge!(g7,1,3,:weight, -0.5*C_BG_Th)
	add_edge!(g7,2,3,:weight, C_BG_Th)
	
	add_edge!(g7,3,4,:weight, -0.5*C_BG_Th)
	
	add_edge!(g7,4,5,:weight, C_BG_Th_Cor)
	add_edge!(g7,6,5,:weight, 6*C_Cor)
	
	add_edge!(g7,5,6,:weight, 4.8*C_Cor)
	add_edge!(g7,7,6,:weight, -1.5*C_Cor)

	add_edge!(g7,6,7,:weight, 1.5*C_Cor)
	add_edge!(g7,7,7,:weight, -3.3*C_Cor)
end

# ╔═╡ 5fc63975-3a15-4430-a7f1-4e5db64c04a1
AdjMatrixfromLinearNeuroGraph(g7)

# ╔═╡ 33002a2b-f8a9-4728-8288-2f92d3b89948
@named seven_regions_gr = ODEfromGraph(g7)

# ╔═╡ c4b4aa78-0324-4ea8-9903-efe87f6074e8
seven_regions_s = structural_simplify(seven_regions_gr)

# ╔═╡ 1a48d894-f43b-4559-8844-50b6e1989bda
sim_dur = 5.0 # Simulate for 10 Seconds

# ╔═╡ 906a3f36-613c-465c-b7d1-6caa245cfe86
prob = ODEProblem(seven_regions_s, [], (0,sim_dur), [])

# ╔═╡ dd455500-d1bf-443d-b589-d400b6844874
prob.p

# ╔═╡ 7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
seven_regions_s.ps

# ╔═╡ cb87381d-330d-4d01-b4f3-55e3a22b2356
typeof(seven_regions_s.ps[6])

# ╔═╡ fe5650bf-a7d3-44a4-b11f-511248c56b0d
@parameters GPe₊H

# ╔═╡ 0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
seven_regions_s.states

# ╔═╡ a1c1b45f-8692-4456-ab20-d72b3e44fc0d
indexof(sym,syms) = findfirst(isequal(sym),syms)

# ╔═╡ f039dc58-04ba-4682-8f87-86ec19bd8d2f
parameters(seven_regions_s)

# ╔═╡ 413e1cd3-00cf-4bb3-98f2-35fa323454cd
begin
	# get the indices of the parameters in the parameter list
	bgth_idx = indexof(C_BG_Th,parameters(seven_regions_s))
	corbgth_idx = indexof(C_Cor_BG_Th,parameters(seven_regions_s))
	cor_idx = indexof(C_Cor,parameters(seven_regions_s))
	bgthcor_idx = indexof(C_BG_Th_Cor,parameters(seven_regions_s))
	gpeh_idx = indexof(GPe₊H,parameters(seven_regions_s))
end

# ╔═╡ c0d99849-ebe8-46b4-8a99-e981a2a4d5dc
gpeh_idx

# ╔═╡ b94c0e24-1793-4fe0-b84d-974d0c27113c
md"""
Cor BG-Th
$(@bind corbgth html"<input type=range min=3 max=20 step=0.05>")

BG-Th Cor
$(@bind bgthcor html"<input type=range min=3 max=20 step=0.05>")

GPeH
$(@bind h html"<input type=range min=0 max=500 step=1>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	# set the parameters for the simulation using the sliders
	p_new = prob.p
	p_new[ corbgth_idx] = corbgth
	p_new[gpeh_idx] = h
	p_new[bgthcor_idx] = bgthcor
	prob_new = remake(prob; p=p_new, u0=ones(14)*0.1)
	sol = solve(prob_new, Rodas4())
end

# ╔═╡ 8accf027-0261-42e5-ac11-c066cfb57c43
begin
	l = @layout [a; b; c]
	p1 = plot(sol.t,sol[9,:],label="EI")
	p2 = plot(sol.t,sol[11,:],label="PY")
	p3 = plot(sol.t,sol[13,:],label="II")
	plot(p1, p2, p3, layout = l)
end

# ╔═╡ c81dfd67-d338-43af-82b4-a83671c3148d
(corbgth,bgthcor,h)

# ╔═╡ bcb92a18-166c-46a7-aace-ccca97a825e4
begin
	l2 = @layout [a b; c d]
	p4 = plot(sol.t,sol[1,:],label="GPe")
	p5 = plot(sol.t,sol[3,:],label="STN")
	p6 = plot(sol.t,sol[5,:],label="GPi")
	p7 = plot(sol.t,sol[7,:],label="Th")
	plot(p4, p5, p6, p7, layout = l2)
end

# ╔═╡ Cell order:
# ╠═771d1460-c48a-11ec-10d4-c7c5dd2a9984
# ╠═66460e85-1544-406b-9c79-773ab174a5cb
# ╠═9aa9ae2b-b8a0-463d-8e9e-b3339b25a99d
# ╠═f45de893-522a-4c0d-b1f0-9093623208ee
# ╠═5fc63975-3a15-4430-a7f1-4e5db64c04a1
# ╠═33002a2b-f8a9-4728-8288-2f92d3b89948
# ╠═c4b4aa78-0324-4ea8-9903-efe87f6074e8
# ╠═1a48d894-f43b-4559-8844-50b6e1989bda
# ╠═906a3f36-613c-465c-b7d1-6caa245cfe86
# ╠═dd455500-d1bf-443d-b589-d400b6844874
# ╠═7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
# ╠═cb87381d-330d-4d01-b4f3-55e3a22b2356
# ╠═fe5650bf-a7d3-44a4-b11f-511248c56b0d
# ╠═0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
# ╠═a1c1b45f-8692-4456-ab20-d72b3e44fc0d
# ╠═f039dc58-04ba-4682-8f87-86ec19bd8d2f
# ╠═413e1cd3-00cf-4bb3-98f2-35fa323454cd
# ╠═c0d99849-ebe8-46b4-8a99-e981a2a4d5dc
# ╠═32d1b83e-acfa-4351-9bcf-bc4367781186
# ╟─8accf027-0261-42e5-ac11-c066cfb57c43
# ╟─b94c0e24-1793-4fe0-b84d-974d0c27113c
# ╟─c81dfd67-d338-43af-82b4-a83671c3148d
# ╠═bcb92a18-166c-46a7-aace-ccca97a825e4
