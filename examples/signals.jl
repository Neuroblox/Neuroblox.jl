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

# ╔═╡ 4111e8d4-c16d-11ec-12d6-210de255a17b
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate()
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("Plots")
	Pkg.develop(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
	Pkg.add("OrdinaryDiffEq")
    using Plots, Neuroblox, OrdinaryDiffEq
end

# ╔═╡ 4e50391b-3601-47a5-8887-0f489117cd43
begin
	time = collect(0:0.01:3)
	phase_int = phase_inter(0:3,[0.0,1.0,2.0,1.0])
	phase_cos_out(ω,t) = phase_cos_blox(ω,t,phase_int)
	phase_sin_out(ω,t) = phase_sin_blox(ω,t,phase_int)
end

# ╔═╡ 41b807c3-c8b9-4f93-86f6-c066f99fdad1
begin
	@named Str2 = jansen_ritC(τ=0.0022, H=20, λ=300, r=0.3)
	@parameters phase_input=0 ampl=1

	sys = [Str2.system]
	eqs = [sys[1].jcn ~ ampl*phase_input]
	@named phase_system = ODESystem(eqs,systems=sys)
	phase_system_simpl = structural_simplify(phase_system)
end

# ╔═╡ 80769855-2c7f-4684-ad29-cd51c0a44bd5
phase_ode = ODAEProblem(phase_system_simpl,[],(0,3.0),[])

# ╔═╡ bfdc6d04-003f-41c2-96ba-026e1dace323
begin
	# create callback functions
	# we always want to update phase_input to be our phase_cos_out(t)
	condition = function (u,t,integrator)
    	true
	end

	function affect!(integrator)
    	integrator.p[1] = phase_cos_out(10*pi,integrator.t)
	end

	cb = DiscreteCallback(condition,affect!)
end

# ╔═╡ e625d6c8-b101-44ce-b094-b5b506339702
phase_ode.p

# ╔═╡ dfa0bc01-aca6-4e1f-b1e2-31c173df072e
sol = solve(phase_ode,Tsit5(),saveat=0.01,callback=cb)

# ╔═╡ 4adf9ac3-d2e4-434d-bcec-d4861e1bc550
plot(time,phase_cos_out.(10*pi,time),label="input")

# ╔═╡ 94bd8b1a-d20e-4266-8e6d-b54bcf5573ef
md"""
ampl
$(@bind a html"<input type=range min=0 max=10000>")
"""

# ╔═╡ 4b8289a8-19e7-4de1-b334-46c12a6d6ba9
begin
	p_new = phase_ode.p
	p_new[1] = phase_cos_out(10*pi,0)
	p_new[2] = a/10000
	prob1 = remake(phase_ode; p=p_new, u0=[0.0,0.0])
	sol2 = solve(prob1,Tsit5(),saveat=0.01,callback=cb)
end

# ╔═╡ b5402079-1780-4e73-9a6f-4156211430eb
plot(sol2.t,sol2[1,:],label="x")

# ╔═╡ 7727fe3e-6dad-47d4-a648-437a8a2e4a86
plot(sol2.t,sol2[2,:],label="y")

# ╔═╡ Cell order:
# ╠═4111e8d4-c16d-11ec-12d6-210de255a17b
# ╠═4e50391b-3601-47a5-8887-0f489117cd43
# ╠═41b807c3-c8b9-4f93-86f6-c066f99fdad1
# ╠═80769855-2c7f-4684-ad29-cd51c0a44bd5
# ╠═bfdc6d04-003f-41c2-96ba-026e1dace323
# ╠═e625d6c8-b101-44ce-b094-b5b506339702
# ╠═dfa0bc01-aca6-4e1f-b1e2-31c173df072e
# ╠═4adf9ac3-d2e4-434d-bcec-d4861e1bc550
# ╠═4b8289a8-19e7-4de1-b334-46c12a6d6ba9
# ╠═b5402079-1780-4e73-9a6f-4156211430eb
# ╟─94bd8b1a-d20e-4266-8e6d-b54bcf5573ef
# ╠═7727fe3e-6dad-47d4-a648-437a8a2e4a86
