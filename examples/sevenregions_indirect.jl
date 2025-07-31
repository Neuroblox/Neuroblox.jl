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
    using Plots, Neuroblox, OrdinaryDiffEq, MetaGraphs, Graphs, DSP, Printf
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
	gpeh_idx = 6
	stnh_idx = 10
end

# ╔═╡ 3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
md"""
GPeH
$(@bind hg html"<input type=range min=0 max=500 step=1>")
STNH
$(@bind hs html"<input type=range min=0 max=300 step=1>")
"""

# ╔═╡ 32d1b83e-acfa-4351-9bcf-bc4367781186
begin
	# set the parameters for the simulation using the sliders
	p_new = prob.p
	p_new[gpeh_idx] = hg
	p_new[stnh_idx] = hs
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

# ╔═╡ 9328ebbd-232f-49a5-8d6c-bf6cae71b898
# Compute GPe and STN Phase Angle
begin
	phase_angle_gpe = Neuroblox.phaseangle(sol[1,:])
	phase_angle_stn = Neuroblox.phaseangle(sol[3,:])
end

# ╔═╡ 8fd5ac10-e306-49a7-aca4-397918dc242d
# Compute GPe and STN PSD
begin
	f_gpe, pxx_gpe = Neuroblox.powerspectrum(sol[1,:], length(sol[1,:]), 1000, "pwelch", hanning)
	f_stn, pxx_stn = Neuroblox.powerspectrum(sol[3,:], length(sol[3,:]), 1000, "pwelch", hanning)
end

# ╔═╡ c81dfd67-d338-43af-82b4-a83671c3148d
(hg, hs)

# ╔═╡ c743e876-aa11-4533-af8d-d2ec8f2bb22f
phase_difference = angle.(exp.(im*phase_angle_gpe)./exp.(im*phase_angle_stn))

# ╔═╡ a5a247b0-89a7-47ae-86dd-9bd6b9156731
begin
	ll = @layout [a b; c d]

	p_lfp = plot(sol.t,sol[1,:],label="GPe", lw=2.5, xlims=(0, 1),color="blue")
	p_lfp = plot!(sol.t,sol[3,:],label="STN", lw=2.5, xlims=(0, 1),color="orange", fg_legend = :false, xlabel="time in sec", ylabel="arb. V")
	title!("Simulated LFP")

	p_dyn = plot(sol[1,:], sol[3,:], label=false, xlabel="GPe", ylabel="STN", color="black")
	title!("Phase Space")
	
	p_phase = plot(sol.t, phase_angle_gpe, xlims=(0.7, 1.0), color="blue", label="GPe", lw=2.5, fg_legend = :false)
	p_phase = plot!(sol.t, phase_angle_stn, xlims=(0.7, 1.0), color="orange", label="STN", lw=2.5, xlabel="time in sec", ylabel="Circular Position")
	title!("Phase Synchrony")

	#peak_value_gpe = f_gpe[findmax(pxx_gpe)[2]]
	#peak_value_stn = f_stn[findmax(pxx_stn)[2]]
	#p_bar = bar(["GPe"], [peak_value_gpe], color="blue", label="GPe")
	#p_bar = bar!(["STN"], [peak_value_stn], color="orange", label="STN")
	#title!("Center Frequency")
	
	p_power = plot(f_gpe, pxx_gpe/1000, xlims=(0,30), lw=2.5, label="GPe",color="blue")
	p_power = plot!(f_stn, 10*pxx_stn/1000, xlims=(0,30), lw=2.5, label="10xSTN",color="orange", fg_legend = :false, xlabel="Hz", ylabel="arb. V/Hz^2")
	title!("Power Spectral Density")

	plot(p_lfp, p_dyn, p_phase, p_power, layout = ll, size=(800,400))

end

# ╔═╡ 35b8f488-372b-44ef-a53b-c5bd26672205
# Make the animation

begin
	hga = 0.01
	animg = @animate while hga<600
		# set the parameters for the simulation using the sliders
		p_an = prob.p
		p_an[gpeh_idx] = hga
		p_an[stnh_idx] = hs
		prob_an = remake(prob; p=p_an, u0=ones(14)*0.1)
		sol_an = solve(prob_an, Rodas4())

		# Synchrony Figure for Stimulating Indirect Pathway Demo
		p_lfpa = plot(sol_an.t, sol_an[1,:],label="GPe H=$(@sprintf("%.2f", hga))", lw=2.5, xlims=(0,1), color="blue", fg_legend = :false)
		p_lfpa = plot!(sol_an.t, sol_an[3,:],label="STN", lw=2.5, xlims=(0,5), xlabel="time in sec", ylabel="arb. V", color="orange")
		title!("Simulated LFP")

		p_dyna = plot(sol_an[1,:], sol_an[3,:], label=false, xlabel="GPe", ylabel="STN", color="black")
		title!("Phase Space")

		phase_angle_gpea = Neuroblox.phaseangle(sol_an[1,:])
		phase_angle_stna = Neuroblox.phaseangle(sol_an[3,:])
		
		p_phasea = plot(sol_an.t, phase_angle_gpea, xlims=(0.7, 1.0), color="blue", label="GPe", lw=2.5, fg_legend = :false)
		p_phasea = plot!(sol_an.t, phase_angle_stna, xlims=(0.7, 1.0), color="orange", label="STN", lw=2.5, xlabel="time in sec", ylabel="Circular Position")
		title!("Phase Synchrony")

		#f_gpe_an, pxx_gpe_an = Neuroblox.powerspectrum(sol_an[1,:],
			#length(sol_an[1,:]), 1000, "pwelch", hanning)
		#f_stn_an, pxx_stn_an = Neuroblox.powerspectrum(sol_an[3,:],
			#length(sol_an[3,:]), 1000, "pwelch", hanning)

		alpha_powera_gpe = Neuroblox.bandpassfilter(sol_an[1,:], 8, 16, 1000, 6)
		f_gpe_an_a, pxx_gpe_an_a = Neuroblox.powerspectrum(alpha_powera_gpe,
			length(alpha_powera_gpe), 1000, "pwelch", hanning)

		beta_powera_gpe = Neuroblox.bandpassfilter(sol_an[1,:], 16, 35, 1000, 6)
		f_gpe_an_b, pxx_gpe_an_b = Neuroblox.powerspectrum(beta_powera_gpe,
			length(beta_powera_gpe), 1000, "pwelch", hanning)

		alpha_powera_stn = Neuroblox.bandpassfilter(sol_an[3,:], 8, 16, 1000, 6)
		f_stn_an_a, pxx_stn_an_a = Neuroblox.powerspectrum(alpha_powera_stn,
			length(alpha_powera_stn), 1000, "pwelch", hanning)

		beta_powera_stn = Neuroblox.bandpassfilter(sol_an[1,:], 16, 35, 1000, 6)
		f_stn_an_b, pxx_stn_an_b = Neuroblox.powerspectrum(beta_powera_stn,
			length(beta_powera_stn), 1000, "pwelch", hanning)
				
		#p_powera = plot(f_gpe_an, pxx_gpe_an/1000, xlims=(0,30), lw=2.5, label="GPe",color="blue")
		#p_powera = plot!(f_stn_an, 10*pxx_stn_an/1000, xlims=(0,30), lw=2.5, label="10xSTN",color="orange", fg_legend = :false, xlabel="Hz", ylabel="arb. V/Hz^2")
		p_powera = bar(["Alpha", "Beta"], [sum(pxx_gpe_an_a)/1000, sum(pxx_gpe_an_b)], label="GPe")
		p_powera = bar!(["Alpha", "Beta"], [sum(pxx_stn_an_a)/1000, sum(pxx_stn_an_b)], label="STN/1000")
		title!("Power Spectral Density")
		
		plot(p_lfpa, p_dyna, p_phasea, p_powera, layout = ll, size=(800,400))
		if hga<0.1
			hga = hga + 0.01
		else
			hga = hga + 10
		end
	end
	gif(animg, "anim_fps15_indirect.gif", fps = 15)
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
# ╠═9328ebbd-232f-49a5-8d6c-bf6cae71b898
# ╠═8fd5ac10-e306-49a7-aca4-397918dc242d
# ╠═3213bf3e-0b6e-477c-bcf9-af8f13cb5dfc
# ╠═c743e876-aa11-4533-af8d-d2ec8f2bb22f
# ╠═a5a247b0-89a7-47ae-86dd-9bd6b9156731
# ╠═35b8f488-372b-44ef-a53b-c5bd26672205
