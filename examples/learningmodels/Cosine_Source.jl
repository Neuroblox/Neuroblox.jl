### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ b0fb6c60-fe0d-11ec-27bc-fd3ebc481638
begin
	import Pkg
	Pkg.develop("Neuroblox")
	Pkg.add("ModelingToolkitStandardLibrary")
	Pkg.add("OrdinaryDiffEq")
	Pkg.add("Plots")
	Pkg.add("Statistics")
	Pkg.add("MAT")
	Pkg.add("DSP")
	Pkg.add("Interpolations")
	Pkg.add("PlutoUI")
end

# ╔═╡ 0944db9c-60ac-498c-8d66-b187d03ec355
using Neuroblox, OrdinaryDiffEq, Plots, Statistics, MAT, DSP, Interpolations, PlutoUI

# ╔═╡ 0f92990f-b71d-4b61-a373-3ebf8aecbb7d
using ModelingToolkitStandardLibrary.Blocks

# ╔═╡ aa825920-ea37-4b25-b866-06307f2ccd9a
begin
	# Create Cortico-Striatal Loop
	ω = 20*(2*pi)
	k = ω^2
	@named Cortical    = harmonic_oscillator(ω=ω, ζ=1.0, k=k, h=30.0)
	@named SubCortical = harmonic_oscillator(ω=ω, ζ=1.0, k=k, h=30.0)
end

# ╔═╡ 4ef9362c-f9dd-4e54-9fe9-1cbfa96f09c7
begin
	@parameters t
	#CosineBlox
	mutable struct CosineBlox
	    amplitude::Num
	    frequency::Num
	    phase::Num
	    connector::Num
	    odesystem::ODESystem
	    function CosineBlox(;name, amplitude=1, frequency=20, phase=0)
	
	        sts    = @variables jcn(t)=0.0 u(t)=0.0
	        params = @parameters amplitude=amplitude frequency=frequency phase=phase
	
	        eqs = [u ~ amplitude * cos(2 * pi * frequency * (t) + phase)]
	        odesys = ODESystem(eqs, t, sts, params; name=name)
	
	        new(amplitude, frequency, phase, odesys.u, odesys)
	    end
	end
	const cosine_source = CosineBlox
end

# ╔═╡ 6ee271d3-4b45-40ae-abca-5a6275b50a47
@named Cortical_Source    = cosine_source(amplitude=100, frequency=20, phase=pi)

# ╔═╡ 656fd1f8-78e5-41ed-a3e3-de17073c1a39
@named SubCortical_Source = cosine_source(amplitude=5, frequency=15, phase=pi/2)

# ╔═╡ 6a8a5568-1d53-4ad0-9d3b-8a3e0c758656
begin
	blox    = [Cortical, SubCortical, Cortical_Source, SubCortical_Source]
	sys     = [s.odesystem for s in blox]
	connect = [s.connector for s in blox]
end

# ╔═╡ ce1e120f-093a-441d-9d3b-7b90fccbd621
@parameters g[1:4, 1:4] = [0 1 0 0;
                           1 0 0 0;
                           1 0 0 0;
						   0 1 0 0] 

# ╔═╡ c9ea64c8-c85c-4774-a006-5202edf6c8bb
begin
	@named Loop = LinearConnections(sys=sys, adj_matrix = g, connector=connect)
	prob = ODEProblem(structural_simplify(Loop), [], (0,10), [])
end

# ╔═╡ b80432b8-00a2-45c0-b035-848931ab13fb
Loop.states

# ╔═╡ afe06bf4-3d4d-4d34-8b93-db64e0eb8333
sol = solve(prob, Rodas4())

# ╔═╡ 2bfa6799-d7fa-4ec5-873f-4ea166a7735b
begin
	plot(sol.t, sol[1,:], label="Cortical", lc=:blue)
	plot!(sol.t, sol[3,:], label="SubCortical", lc=:orange, xlims=(0,5), lw=:2.0, xlabel="Time (s)", ylabel="V Units", title="Simulated LFP Signals")
end

# ╔═╡ Cell order:
# ╠═b0fb6c60-fe0d-11ec-27bc-fd3ebc481638
# ╠═0944db9c-60ac-498c-8d66-b187d03ec355
# ╠═0f92990f-b71d-4b61-a373-3ebf8aecbb7d
# ╠═aa825920-ea37-4b25-b866-06307f2ccd9a
# ╠═4ef9362c-f9dd-4e54-9fe9-1cbfa96f09c7
# ╠═6ee271d3-4b45-40ae-abca-5a6275b50a47
# ╠═656fd1f8-78e5-41ed-a3e3-de17073c1a39
# ╠═6a8a5568-1d53-4ad0-9d3b-8a3e0c758656
# ╠═ce1e120f-093a-441d-9d3b-7b90fccbd621
# ╠═c9ea64c8-c85c-4774-a006-5202edf6c8bb
# ╠═b80432b8-00a2-45c0-b035-848931ab13fb
# ╠═afe06bf4-3d4d-4d34-8b93-db64e0eb8333
# ╠═2bfa6799-d7fa-4ec5-873f-4ea166a7735b
