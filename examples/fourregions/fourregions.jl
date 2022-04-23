### A Pluto.jl notebook ###
# v0.19.2

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

# ╔═╡ 0e0de3b2-bb33-11ec-398c-8bb0b5b319bb
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate()
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("Plots")
	Pkg.add(path=joinpath(@__DIR__, "..", "..", "..", "Neuroblox.jl"))
	Pkg.add(["Graphs","MetaGraphs"])
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, Graphs, MetaGraphs, OrdinaryDiffEq
end

# ╔═╡ f5dd2f05-3e53-49fb-9e38-2e570d89979a
begin
	@named Str = jansen_rit(τ=0.0022, H=20, λ=300, r=0.3)
	@named GPe = jansen_rit(τ=0.04, H=20, λ=400, r=0.1)
	@named GPi = harmonic_oscillator(ω=18*2*π, ζ=1, k=(18*2*π)^2, h=0.1)
	@named STN = harmonic_oscillator(ω=18*2*π, ζ=1, k=(18*2*π)^2, h=1.0)
end

# ╔═╡ 696655d0-0f58-44d6-84a6-1142ad5e1021
begin
	@parameters C_1=2.1 C_2=1.1 C_3=1.11 C_4=1.111 C_5=1.1111 C_6=1.11111 C_7=1.2 C_8=1.3
end

# ╔═╡ 72ee87b4-34c5-4252-9c1d-af0c7d30a6dc
begin
	g = LinearNeuroGraph(MetaDiGraph())
	add_blox!(g,Str)
	add_blox!(g,GPe)
	# Changed the order of the blox to match the order they are named for clarity on the adjacency matrix
	add_blox!(g,GPi)
	add_blox!(g,STN)

	#add_edge!(g,3,1,:weight,C_1) # GPi to STR should not exist, removed	
	#add_edge!(g,2,1,:weight,C_2) # GPe to STR should not exist, removed
	#add_edge!(g,3,4,:weight,C_2) # GPi to STN should not exist, removed
	
	# C_1 and C_2 should both be inhibitory
	add_edge!(g,1,3,:weight,C_1)  # STR to GPi 
	add_edge!(g,1,2,:weight,C_2)  # STR to GPe 

	# C_3 should be inhibitory
	add_edge!(g,2,4,:weight,C_3)  # GPe to STN 
	
	# add self-inhibition of GPe, C_4 should be inhibitory
	add_edge!(g,2,2,:weight,C_4)  # GPe to GPe
	
	# add GPe to GPi connection, C_5 should be inhibitory
	add_edge!(g,2,3,:weight,C_5) # GPe to GPi
	
	# add STN to GPi connection, C_6 should be excitatory
	add_edge!(g,4,3,:weight,C_6) # STN to GPi
	
	# C_7 should be excitatory
	add_edge!(g,4,2,:weight,C_7)  # STN to GPe 

	# There should be no inputs to the Striatum from GPe, GPi, or STN. 
	# However, GPi  connects Thalamus->Cortex->Striatum, so let's create a "connection" from GPi to STR to close this loop and play around with mixed blox types. 
	# This could be net excitatory or net inhibitory depending on the circumstances
	add_edge!(g,3,1,:weight,C_8) # GPi to STR 
end

# ╔═╡ 13566bad-0da0-40b4-831d-0cbd1de31b13
adj = AdjMatrixfromLinearNeuroGraph(g)

# ╔═╡ 12e9b29e-48ac-4cad-b732-c08f9ecf03e9
@named four_regions_gr = ODEfromGraph(g=g)

# ╔═╡ 477619d7-4c2a-4b95-8934-c8efd810b67b
Circuit_ss = structural_simplify(four_regions_gr)

# ╔═╡ 6081c9f0-6272-4127-9825-547814132e86
sim_dur = 10.0 # Simulate for 10 Seconds

# ╔═╡ 85bf2521-ffb0-42ed-9e3a-fe7b56805c23
prob = ODAEProblem(Circuit_ss, [], sim_dur, [])

# ╔═╡ 687da2ca-392b-4357-ba17-dec7ce5cd7b8
Circuit_ss.ps

# ╔═╡ e453027b-49aa-468a-a4c9-3d93a69f2b32
prob.p

# ╔═╡ ce3606c0-0c15-456b-bf02-8995ab84d707
md"""
$(@bind c1 html"<input type=range min=0 max=10000>")
$(@bind c2 html"<input type=range min=0 max=10000>")
$(@bind c3 html"<input type=range min=0 max=10000>")
$(@bind c4 html"<input type=range min=0 max=10000>")
$(@bind c5 html"<input type=range min=0 max=10000>")
$(@bind c6 html"<input type=range min=0 max=10000>")
$(@bind c7 html"<input type=range min=0 max=10000>")
$(@bind c8 html"<input type=range min=0 max=10000>")
"""

# ╔═╡ 4acd166f-7d6d-4724-8232-63a4d839ec6a
begin
	para2 = prob.p
	para2[2] = (c1-5000)/1000
	para2[1] = (c2-5000)/1000
	para2[3] = (c3-5000)/1000
	para2[5] = (c4-5000)/1000
	para2[4] = (c5-5000)/1000
	para2[8] = (c6-5000)/1000
	para2[7] = (c7-5000)/1000
	para2[6] = (c8-5000)/1000
	prob2 = remake(prob; p=para2)
	sol = solve(prob2,Tsit5())
end

# ╔═╡ 1ef9cf88-2b14-4c05-ad45-c06c99b83264
begin
	plot(sol.t,sol[1,:])
	plot!(sol.t,sol[2,:])
	plot!(sol.t,sol[3,:])
	plot!(sol.t,sol[4,:])
	
end

# ╔═╡ d73793d3-7a31-44e1-b3f1-b122a8d58d8a
(c1-5000)/1000

# ╔═╡ 76ef9320-63c5-4884-9cc7-bef5de1081ac
md"""
$((c1-5000)/1000,(c2-5000)/1000,(c3-5000)/1000)
"""

# ╔═╡ ce6b0052-0810-4ac9-9f9d-ea4b2f20c8cd
begin
	plot(sol.t,sol[5,:])
	plot!(sol.t,sol[6,:])
	plot!(sol.t,sol[7,:])
	plot!(sol.t,sol[8,:])
end

# ╔═╡ Cell order:
# ╠═0e0de3b2-bb33-11ec-398c-8bb0b5b319bb
# ╠═f5dd2f05-3e53-49fb-9e38-2e570d89979a
# ╠═696655d0-0f58-44d6-84a6-1142ad5e1021
# ╠═72ee87b4-34c5-4252-9c1d-af0c7d30a6dc
# ╠═13566bad-0da0-40b4-831d-0cbd1de31b13
# ╠═12e9b29e-48ac-4cad-b732-c08f9ecf03e9
# ╠═477619d7-4c2a-4b95-8934-c8efd810b67b
# ╠═6081c9f0-6272-4127-9825-547814132e86
# ╠═85bf2521-ffb0-42ed-9e3a-fe7b56805c23
# ╠═687da2ca-392b-4357-ba17-dec7ce5cd7b8
# ╠═e453027b-49aa-468a-a4c9-3d93a69f2b32
# ╠═4acd166f-7d6d-4724-8232-63a4d839ec6a
# ╠═d73793d3-7a31-44e1-b3f1-b122a8d58d8a
# ╟─1ef9cf88-2b14-4c05-ad45-c06c99b83264
# ╟─ce3606c0-0c15-456b-bf02-8995ab84d707
# ╟─76ef9320-63c5-4884-9cc7-bef5de1081ac
# ╠═ce6b0052-0810-4ac9-9f9d-ea4b2f20c8cd
