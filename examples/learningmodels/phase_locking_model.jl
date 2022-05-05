### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ 8d65accf-023f-4e98-adaa-1a1584f770f1
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(@__DIR__)
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("Plots")
	Pkg.add("DSP")
	Pkg.add("MAT")
	Pkg.add("Statistics")
	Pkg.add(path=joinpath(@__DIR__, "..", "..", "Neuroblox.jl"))
    using Plots, Neuroblox, Statistics, MAT, DSP
end

# ╔═╡ 0b6e0e97-27a2-49e9-8ee5-c7c8d0ca991f
begin
	tmin = 0.0
	tmax = 100.0
	dt = 0.001
	T = tmin:dt:tmax
end

# ╔═╡ b0e69b8e-2848-4641-8e0a-7587f7fd20f4
begin
	# Model Parameters
	a_s = 10
	a_p = 10
	ω_s = 18*(2*pi)
	k_s = ω_s^2
	ω_p = 18*(2*pi)
	k_p = ω_p^2
	h = 35.0
end

# ╔═╡ 7b73249c-9254-4c3a-a5a4-20fd7ca310f2
begin
	# Phase Parameters
	phi = matread("phase_data_striatum.mat")
	phi_s = phi["phase_data"]
	phi = matread("phase_data_pfc.mat")
	phi_p = phi["phase_data"]
end

# ╔═╡ 6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
begin
	# Control Parameters
	g0   = 1.0
	g_sp = g0*ones(length(T))
	g_ps = g0*ones(length(T)) 
	Kp_s   = 1.0
	Kp_p   = 1.0
	Ki_s   = 1.0
	Ki_p   = 1.0
	call_rate = Int(0.15/dt)
	controller_call_times = call_rate:call_rate:length(T)
end

# ╔═╡ b7887fa0-bfd8-11ec-32cf-45269db719a2
begin
	
x_s = Array{Float64}(undef, length(T))
y_s = Array{Float64}(undef, length(T))
x_p = Array{Float64}(undef, length(T))
y_p = Array{Float64}(undef, length(T))
x_s[1] = 0.1
y_s[1] = 0.1
x_p[1] = 0.1
y_p[1] = 0.1

phi_x_s = []
phi_x_p = []	
circular_difference = []
	
control_error_s_p = []
control_error_p_s = []
cumulative_error_s_p = []
cumulative_error_p_s = []
cumulative_error_all = []
	
for t = 1:length(T)-1

    if t in controller_call_times[1:length(controller_call_times)-1]
		
		curr = Int(t/call_rate)

		x_s_filt = Neuroblox.bandpassfilter(x_s[(t+1-call_rate):t], 15, 21, 1/dt, 6)
		push!(phi_x_s, angle.(hilbert(x_s_filt)))
		x_p_filt = Neuroblox.bandpassfilter(x_p[(t+1-call_rate):t], 15, 21, 1/dt, 6)
        push!(phi_x_p, angle.(hilbert(x_p_filt)))

		diff = exp.(im.*phi_x_s[curr]) ./ exp.(im.*phi_x_p[curr])
        push!(circular_difference, angle.(diff))
		if mean(circular_difference[curr]) > 0
			push!(control_error_s_p, mean(circular_difference[curr]))
			push!(cumulative_error_s_p, cumsum(control_error_s_p))	
			push!(control_error_p_s, 0)
			push!(cumulative_error_p_s, cumsum(control_error_p_s))
		end
		if mean(circular_difference[curr]) < 0
			push!(control_error_p_s, mean(circular_difference[curr]))
			push!(cumulative_error_p_s, cumsum(control_error_p_s))
			push!(control_error_s_p, 0)
			push!(cumulative_error_s_p, cumsum(control_error_s_p))
		end

		cumulative_error = last(cumulative_error_s_p[length(cumulative_error_s_p)]) + last(cumulative_error_p_s[length(cumulative_error_p_s)])
		push!(cumulative_error_all, cumulative_error)

		if mean(circular_difference[curr]) > 0
			g_sp[t:t+call_rate] .= g0 .+ Kp_s.*(abs(control_error_s_p[curr])) .+ Ki_s.*(cumulative_error)
			g_ps[t:t+call_rate] .= g_ps[t-1] 
		end

		if mean(circular_difference[curr]) < 0
			g_sp[t:t+call_rate] .= g_sp[t-1]
			g_ps[t:t+call_rate] .= g0 .+ Kp_p.*(abs(control_error_p_s[curr])) .+ Ki_p.*(cumulative_error)
		end
		
		if mean(circular_difference[curr]) == 0
			g_sp[t:t+call_rate] .= g_sp[t-1]
			g_ps[t:t+call_rate] .= g_ps[t-1]
		end

    end

    dx_s = y_s[t]-(2ω_s*x_s[t])+ k_s*(2/π)*atan(g_ps[t]*(x_p[t]/h) + a_s*cos(ω_s*t +      phi_s[t]))
    dy_s = -(ω_s^2)*x_s[t]	
    dx_p = y_p[t]-(2*ω_p*x_p[t])+ k_p*(2/π)*atan(g_sp[t]*(x_s[t]/h) + a_p*cos(ω_p*t +     phi_p[t]))
    dy_p = -(ω_p^2)*x_p[t]

	#RK4 Application	
	k1_xs = dt*dx_s
	k2_xs = (dt/2)*(dx_s + k1_xs/2)
	k3_xs = (dt/2)*(dx_s + k2_xs/2)
	k4_xs = (dt/2)*(dx_s + k3_xs/2)
    x_s[t+1] = x_s[t] + (k1_xs + 2*k2_xs + 2*k3_xs + k4_xs)/6
		
	k1_ys = dt*dy_s
	k2_ys = (dt/2)*(dy_s + k1_ys/2)
	k3_ys = (dt/2)*(dy_s + k2_ys/2)
	k4_ys = (dt/2)*(dy_s + k3_ys/2)
    y_s[t+1] = y_s[t] + (k1_ys + 2*k2_ys + 2*k3_ys + k4_ys)/6
		
	k1_xp = dt*dx_p
	k2_xp = (dt/2)*(dx_p + k1_xp/2)
	k3_xp = (dt/2)*(dx_p + k2_xp/2)
	k4_xp = (dt/2)*(dx_p + k3_xp/2)
    x_p[t+1] = x_p[t] + (k1_xp + 2*k2_xp + 2*k3_xp + k4_xp)/6
		
	k1_yp = dt*dy_p
	k2_yp = (dt/2)*(dy_p + k1_yp/2)
	k3_yp = (dt/2)*(dy_p + k2_yp/2)
	k4_yp = (dt/2)*(dy_p + k3_yp/2)
    y_p[t+1] = y_p[t] + (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)/6
			
end
end

# ╔═╡ 1f9a8e9a-b825-459a-9de0-ab884c12f06f
begin
	
	l = @layout [a b; c d]

	p1 = plot(T, x_s, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=4.0,  lc=:red, xlims=(0.5,1.5), xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	p1 = plot!(T, x_p, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(0.5,1.5), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10, ylims=(-85,85))
	title!("Earlier Learning (Unfiltered LFP Signal)")

	x_s_filt_all = Neuroblox.bandpassfilter(x_s, 15, 21, 1/dt, 6)
	x_p_filt_all = Neuroblox.bandpassfilter(x_p, 15, 21, 1/dt, 6)
	
	p2 = plot(T, x_s_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=4.0,  lc=:red, xlims=(0.5,1.5), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	p2 = plot!(T, x_p_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(0.5,1.5), ylims=(-85,85), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	title!("Earlier Learning (Filtered LFP Signal)")

	p3 = plot(T, x_s, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=4.0, lc=:red, xlims=(95.5, 96.5), ylims=(-85,85), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	p3 = plot!(T, x_p, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(95.5, 96.5), ylims=(-85,85), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	title!("Later Learning (Unfiltered LFP Signal)")

	p4 = plot(T, x_s_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=3.5, lc=:red, xlims=(95.5,96.5), ylims=(-85,85), fg_legend = :false, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)	
	p4 = plot!(T, x_p_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(95.5,96.5), ylims=(-85,85), fg_legend = :false, legend = :outertop, xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=10)
	title!("Later Learning (Filtered LFP Signal)")

	plot(p1, p2, p3, p4, layout = l, size=(900,650))

end

# ╔═╡ b7d1d644-3412-4b0e-b504-41d9dfc8147a
begin

	lphase = @layout [a b; c d]
	
	x_s_grab = Neuroblox.bandpassfilter(x_s, 15, 21, 1/dt, 6)
	x_s_angle = Neuroblox.phaseangle(x_s_grab)
	x_p_grab = Neuroblox.bandpassfilter(x_p, 15, 21, 1/dt, 6)
	x_p_angle = Neuroblox.phaseangle(x_p_grab)
	
	ppe = plot(T, x_s_angle, label="phi_str", tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, lc=:red, xlims=(1.5,2), lw=2.0)
	ppe = plot!(T, x_p_angle, label = "phi_pfc", tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1.5,2), lw=2.0)
	title!("Earlier Learning (Phase)")

	ppl = plot(T, x_s_angle, label="phi_str", tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, lc=:red, xlims=(96.5,97), lw=2.0)
	ppl = plot!(T, x_p_angle, label = "phi_pfc", tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, fg_legend = :false, legend = :outertop, lc=:black, xlims=(96.5,97), lw=2.0)
	title!("Later Learning (Phase)")

	circular_diff_all = angle.(exp.(im*x_s_angle)./exp.(im*x_p_angle))
	pcp = plot(T, circular_diff_all, label="circular phase difference", tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, fg_legend = :false)
	title!("Circular Phase Difference")

	pe = plot(control_error_s_p, label="err_sp(t)", linecolor = "red", lw=3.0)
	pe = plot!(control_error_p_s, label="err_ps(t)", linecolor ="black", lw=:2.0, fg_legend = :false, legend = :outertop, tickfontsize=11,ytickfontsize=11, xguidefontsize=11, yguidefontsize=11,legendfontsize=11, xlabel = "Controller Call #", ylabel="Error Signal")
	title!("Control Error")

	plot(ppe, ppl, pcp, pe, layout = lphase, size=[700, 600])
	
end

# ╔═╡ d577e2b9-59fb-4a26-9b4e-ee9952cb364c
begin
	lll = @layout [a b]
	p7 = plot(cumulative_error_s_p, lw=5.0, lc=:red, title="Cumulative Control Error (S->P)", xlabel="Call #", ylabel="Cumulative Error", label=false, ylims=(0,100)) 
	p8 = plot(cumulative_error_p_s, lw=5.0, lc=:black, title="Cumulative Control Error (P->S)", xlabel="Call #", ylabel="Cumulative Error", label=false, ylims=(-100,0)) 
	plot(p7, p8, layout=lll, size=[800,500])
end

# ╔═╡ e850f63f-7423-4fc0-a0f4-0f1c0b4eeea3
begin
	plot(T, g_sp, label="g (s->p)", lw=3.0, lc=:red, fg_legend = :false, legend = :outertop, xtickfontsize=14,ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,legendfontsize=14, xlims=(0,99.8))
	plot!(T, g_ps, label="g(p->s)", lw=2.7, lc=:black, fg_legend = :false, legend = :outertop, xtickfontsize=14,ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,legendfontsize=14, xlims=(0,99.8), size=(800,500))
	title!("Proportional-Integral Controller")
	xlabel!("Time (s)")
	ylabel!("g(t)")
end

# ╔═╡ 82f328fd-45fa-4dff-9f63-20f71ac82949
begin
	f_s, power_s = Neuroblox.powerspectrum(x_s, length(x_s), 1/dt, "pwelch", hanning)
	f_p, power_p = Neuroblox.powerspectrum(x_p, length(x_p), 1/dt, "pwelch", hanning)

	plot(f_s, power_s, label="Striatum", lw=3.0, lc=:red)
	plot!(f_p, power_p, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", 	  ylabel="arb. V/Hz^2", lc=:black, xtickfontsize=14,ytickfontsize=14, xguidefontsize=14, yguidefontsize=14,legendfontsize=14, fg_legend = :false)
	xlims!(0,35)
	title!("Power Spectral Density")
end

# ╔═╡ 71adaea2-c17f-4755-b945-a26e3a29d749
# ╠═╡ disabled = true
#=╠═╡
begin
	
	lfilt = @layout [a;b;c]

	p9 = plot(T, x_s_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=4.0,  lc=:red, xtickfontsize=9,ytickfontsize=9,xguidefontsize=10, yguidefontsize=10)
	p9 = plot!(T, x_p_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(1.05-0.15, 1.05), fg_legend = :false, legend = :outertop, xtickfontsize=9,ytickfontsize=9, xguidefontsize=10, yguidefontsize=10,legendfontsize=10)
	title!("Earlier Learning (Filtered LFP Time Series)")

	p10 = plot(T, x_s_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=4.0, lc=:red,xtickfontsize=9,ytickfontsize=9,xguidefontsize=10, yguidefontsize=10)
	p10 = plot!(T, x_p_filt_all, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=3.5, lc=:black, xlims=(96-0.15,96), fg_legend = :false, legend = :outertop, xtickfontsize=9,ytickfontsize=9, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, ylims=(-15,15))
	title!("Later Learning (Filtered LFP Time Series)")

	p11 = plot([0:1:149]*0.001, control_error[7], label="CE Bin (Earlier)", lw=2.0, lc=:purple, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10)
	p11 = plot!([0:1:149]*0.001, control_error[640], label="CE Bin (Later)", lw=2.0, lc=:blue, fg_legend = :false, legend = :outertop, xtickfontsize=9,ytickfontsize=9, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, ylabel="e(t)", xlims=(0,0.15), ylims=(-pi, pi))
	title!("Control Error")
	xlabel!("Time (s)")
	p11 = hspan!([-0.01, 0.01], label="CE = 0", color="gold", linestyle=:dot, lw=2.0)

	plot(p9, p10, p11, layout = lfilt, size=(650,850))

end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8d65accf-023f-4e98-adaa-1a1584f770f1
# ╠═0b6e0e97-27a2-49e9-8ee5-c7c8d0ca991f
# ╠═b0e69b8e-2848-4641-8e0a-7587f7fd20f4
# ╠═7b73249c-9254-4c3a-a5a4-20fd7ca310f2
# ╠═6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
# ╠═b7887fa0-bfd8-11ec-32cf-45269db719a2
# ╠═1f9a8e9a-b825-459a-9de0-ab884c12f06f
# ╠═b7d1d644-3412-4b0e-b504-41d9dfc8147a
# ╠═d577e2b9-59fb-4a26-9b4e-ee9952cb364c
# ╠═e850f63f-7423-4fc0-a0f4-0f1c0b4eeea3
# ╠═82f328fd-45fa-4dff-9f63-20f71ac82949
# ╠═71adaea2-c17f-4755-b945-a26e3a29d749
