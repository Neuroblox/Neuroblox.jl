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

# ╔═╡ 0472d200-c445-11ec-17d6-bfda34ad541d
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate()
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("Plots")
	Pkg.develop(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs
end

# ╔═╡ 85d06c3c-86e7-4d19-9b64-401e31d2d085
begin
	# Create Regions
	@named GPe       = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
	@named STN       = jansen_ritC(τ=0.01, H=20, λ=500, r=0.1)
	@named GPi       = jansen_ritC(τ=0.014, H=20, λ=400, r=0.1)
	@named Thalamus  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
	@named PFC       = jansen_ritSC(τ=0.001, H=20, λ=5, r=0.15)
end

# ╔═╡ 1d5e080c-2234-45f4-940c-de15b2403a5d
# Connect Regions through Adjacency Matrix
@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

# ╔═╡ 9fb774a7-86d6-4fd8-90e9-0c0b264f497d
begin
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
end

# ╔═╡ eee57334-c755-43dd-b5f4-7806241f44e1
@named five_regions_gr = ODEfromGraph(g=g)

# ╔═╡ b92d05f0-8148-4184-a8af-2fe868428223
five_regions_s = structural_simplify(five_regions_gr)

# ╔═╡ 5246201a-d8aa-4f70-ae15-5533e31b7722
sim_dur = 10.0 # Simulate for 10 Seconds

# ╔═╡ 03a27652-7889-41e6-afde-93d4fef6d2fb
prob = ODAEProblem(five_regions_s, [], (0,sim_dur), [])

# ╔═╡ 26c1a0f1-25fa-4b57-8e7e-a0bbad71d707
five_regions_s.ps

# ╔═╡ e7b74e94-107f-479f-9233-5444417891bd
md"""
C_BG_Th
$(@bind bgth html"<input type=range min=0 max=10000>")
C_Cor_BG_Th
$(@bind corbgth html"<input type=range min=0 max=10000>")
C_BG_Th_Cor
$(@bind bgthcor html"<input type=range min=0 max=10000>")
"""

# ╔═╡ 38f49443-79b7-4f83-b05c-4e2ebf9c53b6
begin
	p_new = prob.p
	p_new[1] = bgth/100
	p_new[2] = (corbgth-5000)/1000
	p_new[3] = (bgthcor-5000)/1000
	prob_new = remake(prob; p=p_new)
	sol = solve(prob_new, Tsit5())
end

# ╔═╡ 9a9a2aed-5671-476b-b285-4edd39e41243
(bgth/100,(corbgth-5000)/1000,(bgthcor-5000)/1000)

# ╔═╡ 05223a1e-f66a-4d54-ab1c-131ebd44c0f8
begin
	plot(sol.t,sol[1,:])
	plot!(sol.t,sol[3,:])
	plot!(sol.t,sol[5,:])
	plot!(sol.t,sol[7,:])
	plot!(sol.t,sol[9,:])
end

# ╔═╡ Cell order:
# ╠═0472d200-c445-11ec-17d6-bfda34ad541d
# ╠═85d06c3c-86e7-4d19-9b64-401e31d2d085
# ╠═1d5e080c-2234-45f4-940c-de15b2403a5d
# ╠═9fb774a7-86d6-4fd8-90e9-0c0b264f497d
# ╠═eee57334-c755-43dd-b5f4-7806241f44e1
# ╠═b92d05f0-8148-4184-a8af-2fe868428223
# ╠═5246201a-d8aa-4f70-ae15-5533e31b7722
# ╠═03a27652-7889-41e6-afde-93d4fef6d2fb
# ╠═26c1a0f1-25fa-4b57-8e7e-a0bbad71d707
# ╠═38f49443-79b7-4f83-b05c-4e2ebf9c53b6
# ╠═e7b74e94-107f-479f-9233-5444417891bd
# ╠═9a9a2aed-5671-476b-b285-4edd39e41243
# ╠═05223a1e-f66a-4d54-ab1c-131ebd44c0f8
