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
	Pkg.add("DSP")
	Pkg.develop(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, DSP, Printf
end

# ╔═╡ 66460e85-1544-406b-9c79-773ab174a5cb
begin
	# Create Regions
	@named WC = wilson_cowan()
end

# ╔═╡ 9f3d6efd-a884-449c-b4b8-42b0b435e245
begin
	sys = [WC.system]
	eqs = [sys[1].jcn ~ 0.0, sys[1].P ~ 0.0]
	@named WC_sys = ODESystem(eqs,systems=sys)
end

# ╔═╡ c4b4aa78-0324-4ea8-9903-efe87f6074e8
WC_s = structural_simplify(WC_sys)

# ╔═╡ 1a48d894-f43b-4559-8844-50b6e1989bda
sim_dur = 100.0 # Simulate for 10 Seconds

# ╔═╡ 906a3f36-613c-465c-b7d1-6caa245cfe86
prob = ODEProblem(WC_s, [], (0,sim_dur), [])

# ╔═╡ dd455500-d1bf-443d-b589-d400b6844874
prob.p

# ╔═╡ 7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
WC_s.ps

# ╔═╡ 0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
WC_s.states

# ╔═╡ a1c1b45f-8692-4456-ab20-d72b3e44fc0d
indexof(sym,syms) = findfirst(isequal(sym),syms)

# ╔═╡ f039dc58-04ba-4682-8f87-86ec19bd8d2f
parameters(WC_s)

# ╔═╡ 3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
md"""
WC₊θ_E
$(@bind tE html"<input type=range value=1.8 min=0 max=50 step=0.1>")
WC₊θ_I
$(@bind tI html"<input type=range value=10.8 min=0 max=50 step=0.1>")
"""

# ╔═╡ e5a045c1-1ada-4726-b7cb-ff011627243d
md"""
WC₊c_EE, 
$(@bind cEE html"<input type=range value=41.2 min=0 max=50 step=0.1>")
WC₊c_IE
$(@bind cIE html"<input type=range value=42.1 min=0 max=50 step=0.1>")
"""

# ╔═╡ 5dc52090-4f13-49d5-b39e-383520a067df
md"""
WC₊c_II, 
$(@bind cII html"<input type=range value=3.2 min=0 max=50 step=0.1>")
WC₊c_EI
$(@bind cEI html"<input type=range value=16.7 min=0 max=50 step=0.1>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	# set the parameters for the simulation using the sliders
	p_new = prob.p
	p_new[3] = cEE
	p_new[4] = cIE
	p_new[5] = cEI
	p_new[6] = cII
	p_new[7] = tE
	p_new[8] = tI
	prob_new = remake(prob; p=p_new)
	sol = solve(prob_new, Rodas4())
end

# ╔═╡ aa9a5f4e-c008-44f3-acd9-37e3df9357ef
begin
	# set the parameters for the simulation using the sliders
	sol2 = solve(prob_new, Euler(),dt=0.01)
end

# ╔═╡ 7530ba11-de39-4d54-af15-7865ce8137b5
tE,tI,cEE,cIE,cII,cEI

# ╔═╡ 8accf027-0261-42e5-ac11-c066cfb57c43
begin
	l = @layout [a; b]
	p1 = plot(sol.t,sol[1,:],label="E")
	p1 = plot!(sol2.t,sol2[1,:],label="E Euler")
	p2 = plot(sol.t,sol[2,:],label="I")
	p2 = plot!(sol2.t,sol2[2,:],label="I Euler")
	plot(p1, p2, layout = l)
end

# ╔═╡ Cell order:
# ╠═771d1460-c48a-11ec-10d4-c7c5dd2a9984
# ╠═66460e85-1544-406b-9c79-773ab174a5cb
# ╠═9f3d6efd-a884-449c-b4b8-42b0b435e245
# ╠═c4b4aa78-0324-4ea8-9903-efe87f6074e8
# ╠═1a48d894-f43b-4559-8844-50b6e1989bda
# ╠═906a3f36-613c-465c-b7d1-6caa245cfe86
# ╠═dd455500-d1bf-443d-b589-d400b6844874
# ╠═7e6ec3b2-a8a7-42a0-87fe-b069c5a66a46
# ╠═0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
# ╠═a1c1b45f-8692-4456-ab20-d72b3e44fc0d
# ╠═f039dc58-04ba-4682-8f87-86ec19bd8d2f
# ╠═32d1b83e-acfa-4351-9bcf-bc4367781186
# ╠═aa9a5f4e-c008-44f3-acd9-37e3df9357ef
# ╟─3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
# ╟─e5a045c1-1ada-4726-b7cb-ff011627243d
# ╟─5dc52090-4f13-49d5-b39e-383520a067df
# ╠═7530ba11-de39-4d54-af15-7865ce8137b5
# ╠═8accf027-0261-42e5-ac11-c066cfb57c43
