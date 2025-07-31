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
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, DSP
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
@named seven_regions_gr = ODEfromGraph(g=g7)

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
	gpeh_idx = 6
end

# ╔═╡ 3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
md"""
GPeH
$(@bind h html"<input type=range min=0 max=500 step=1>")
"""

# ╔═╡ ed02b30f-6ebf-4796-b778-5347fce35dc1
md"""
Cor BG-Th
$(@bind corbgth html"<input type=range min=3 max=300 step=1>")

BG-Th Cor
$(@bind bgthcor html"<input type=range min=3 max=300 step=1>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	# set the parameters for the simulation using the sliders
	p_new = prob.p
	p_new[ corbgth_idx] = corbgth
	p_new[bgthcor_idx] = bgthcor
	p_new[gpeh_idx] = h
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

# ╔═╡ bcb92a18-166c-46a7-aace-ccca97a825e4
begin
	l2 = @layout [a b; c d]
	p4 = plot(sol.t,sol[1,:],label="GPe")
	p5 = plot(sol.t,sol[3,:],label="STN")
	p6 = plot(sol.t,sol[5,:],label="GPi")
	p7 = plot(sol.t,sol[7,:],label="Th")
	plot(p4, p5, p6, p7, layout = l2)
end

# ╔═╡ d9181a23-46a4-4f95-be4e-9b4fc184f1d8
begin
	
	f_gpe, pxx_gpe = Neuroblox.powerspectrum(sol[1,:], length(sol[1,:]), 1000, "pwelch", hanning)
	f_stn, pxx_stn = Neuroblox.powerspectrum(sol[3,:], length(sol[3,:]), 1000, "pwelch", hanning)
	
	# Synchrony Figure for Demo 1
	l22 = @layout [a; b; c]
	p44 = plot(sol.t,sol[1,:],label="GPe", lw=2.0, xlims=(0.5,2.5),color="blue")
	p55 = plot(sol.t,sol[3,:],label="STN", lw=2.0, xlims=(0.5,2.5),xlabel="t in sec",color="orange")
	p66 = plot(f_gpe, pxx_gpe/1000, xlims=(0,30), lw=2.0, label="GPe",color="blue")
	p66 = plot!(f_stn, 10*pxx_stn/1000, xlims=(0,30), lw=2.0, label="10xSTN",xlabel="f in Hz",ylabel="PSD",color="orange")
	plot(p44, p55, p66, layout = l22)
	
end

# ╔═╡ c81dfd67-d338-43af-82b4-a83671c3148d
(corbgth, bgthcor, h)

# ╔═╡ 37bd5c95-7a4f-406e-a11c-bdc273482f68
begin
	f_th, pxx_th = Neuroblox.powerspectrum(sol[7,:], length(sol[7,:]), 1000, "pwelch", hanning)
	f_ei, pxx_ei = Neuroblox.powerspectrum(sol[9,:], length(sol[9,:]), 1000, "pwelch", hanning)
	f_py, pxx_py = Neuroblox.powerspectrum(sol[11,:], length(sol[11,:]), 1000, "pwelch", hanning)

	phase_angle_thalamus = Neuroblox.phaseangle(sol[7,:])
	phase_angle_ei = Neuroblox.phaseangle(sol[9,:])
	phase_angle_py = Neuroblox.phaseangle(sol[11,:])
	phase_angle_stn = Neuroblox.phaseangle(sol[3,:])

	# Synchrony Figure for Demos 2/3
	l_p = @layout [a b;c d]
	
	p_sc = plot(phase_angle_thalamus, label="Th", lw=2.0)
	p_sc = plot!(phase_angle_ei, xlims=(2000,2280), label="EI", lw=2.0)
	title!("Subcortical -> Cortical")

	p_cs = plot(phase_angle_py, label="PY", lw=2.0)
	p_cs = plot!(phase_angle_stn, xlims=(2000,2280), label="STN", lw=2.0)
	title!("Cortical -> Subcortical" )

	power_sc = plot(f_ei, pxx_ei, xlims=(0,60), lw=2.0, label="EI", title="Cortical Input Region", lc=:black, ylabel="PSD", xlabel="Freq (Hz)")
	power_cs = plot(f_stn, pxx_stn, xlims=(0,60), lw=2.0, label="STN", title="Subcortical Input Region", lc=:black, ylabel="PSD", xlabel="Freq (Hz)")

	plot(power_sc, p_sc, power_cs, p_cs, layout = l_p)

end

# ╔═╡ 3d1bb45b-4dc7-45df-a551-6064c1ef4e3e
begin
	anim3 = @animate for hh ∈ 1:10:600
		# set the parameters for the simulation using the sliders
		p_an = prob.p
		p_an[ corbgth_idx] = corbgth
		p_an[gpeh_idx] = hh
		p_an[bgthcor_idx] = bgthcor
		prob_an = remake(prob; p=p_an, u0=ones(14)*0.1)
		sol_an = solve(prob_an, Rodas4())
		
    	f_gpe_an, pxx_gpe_an = Neuroblox.powerspectrum(sol_an[1,:],
			length(sol_an[1,:]), 1000, "pwelch", hanning)
		f_stn_an, pxx_stn_an = Neuroblox.powerspectrum(sol_an[3,:],
			length(sol_an[3,:]), 1000, "pwelch", hanning)

		# Synchrony Figure for Demo 1
		p4a = plot(sol_an.t,sol_an[1,:],label="GPe H=$hh", lw=2.0, xlims=(0.5,2.5),color="blue")
		p5a = plot(sol_an.t,sol_an[3,:],label="STN", lw=2.0, xlims=(0.5,2.5),xlabel="t in sec",color="orange")
		p6a = plot(f_gpe_an, pxx_gpe_an/1000, xlims=(0,30), lw=2.0, label="GPe",color="blue")
		p6a = plot!(f_stn_an, 10*pxx_stn_an/1000, xlims=(0,30),lw=2.0, label="STN",xlabel="f in Hz",ylabel="PSD",color="orange")
		plot(p4a, p5a, p6a, layout = l22)
	end
	gif(anim3, "anim_fps15.gif", fps = 15)
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
# ╠═0447ca7d-9dc0-4777-b4ac-adc2eb7d3c8a
# ╠═a1c1b45f-8692-4456-ab20-d72b3e44fc0d
# ╠═f039dc58-04ba-4682-8f87-86ec19bd8d2f
# ╠═413e1cd3-00cf-4bb3-98f2-35fa323454cd
# ╠═32d1b83e-acfa-4351-9bcf-bc4367781186
# ╠═8accf027-0261-42e5-ac11-c066cfb57c43
# ╠═c81dfd67-d338-43af-82b4-a83671c3148d
# ╠═bcb92a18-166c-46a7-aace-ccca97a825e4
# ╠═3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
# ╠═d9181a23-46a4-4f95-be4e-9b4fc184f1d8
# ╠═ed02b30f-6ebf-4796-b778-5347fce35dc1
# ╠═37bd5c95-7a4f-406e-a11c-bdc273482f68
# ╠═3d1bb45b-4dc7-45df-a551-6064c1ef4e3e
