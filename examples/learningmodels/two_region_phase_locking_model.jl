### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ 93e79dba-82a2-4574-a58f-cb1199fe254b
begin
	import Pkg
	Pkg.add("Plots")
	Pkg.add("Statistics")
	Pkg.add("MAT")
	Pkg.add("DSP")
	Pkg.add("PlutoUI")
	Pkg.develop("Neuroblox")
end

# ╔═╡ 8d65accf-023f-4e98-adaa-1a1584f770f1
using Plots, Statistics, MAT, DSP, Neuroblox, PlutoUI

# ╔═╡ 7b73249c-9254-4c3a-a5a4-20fd7ca310f2
begin
	# Phase Parameters
	phi = matread("phase_subcortical.mat")
	phi_s = phi["phase_str"]
	phi = matread("phase_cortical.mat")
	phi_p = phi["phase_pfc"]
end

# ╔═╡ fd88ae6a-a3e6-46c4-a0d9-c8af6eb92dd5
begin
	tmin = 0.0
	dt = 0.001
	tmax = length(phi_s)*dt
	T = tmin:dt:tmax
end

# ╔═╡ 2d79cfd8-0ca1-418a-9065-6ac15764d972
begin
	# Model Parameters
	a_s = 10
	a_p = 10
	ω_s = 30*(2*pi)
	k_s = ω_s^2
	ω_p = 30*(2*pi)
	k_p = ω_p^2
	h = 35.0
end

# ╔═╡ 6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
begin
	# Control Parameters
	g0   = 1.0
	g_sp = g0*ones(length(T))
	g_ps = g0*ones(length(T)) 
	Kp_s   = 0.1#1
	Kp_p   = 0.1#1
	Ki_s   = 0.1#1
	Ki_p   = 0.1#1
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

# ╔═╡ fddaed52-854d-4f5d-b2d4-f390ee74fa8a
begin

	fig1 = @layout [a; b]
	
	p_ce = plot(control_error_s_p, label="e(S->P)", linecolor = "red", lw=2.0, fg_legend = :false, legend = :bottomright, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100), ylims=(-3.14,3.14))
	p_ce = plot!(control_error_p_s, label="e(P->S)", linecolor ="black", lw=:2.0, fg_legend = :false, legend = :bottomright, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100), grid = false, ylims=(-3.14,3.14), xticks = 0:750:4500, titlefontsize=13)
	title!("Control Error")
	
	p_gain = plot(T, g_sp, label="g(S->P)", lw=2.5, lc=:red, fg_legend = :false, legend = :bottomright, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10)
	p_gain = plot!(T, g_ps, label="g(P->S)", lw=2.3, lc=:black, fg_legend = :false, legend = :bottomright, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, xlims=(0, tmax-1), grid = false, xticks = 0:75:725, titlefontsize=13)
	title!("Reinforce Plasticity")
	xlabel!("Time (s)")
	ylabel!("Edge Weights")

	plot(p_ce, p_gain, layout = fig1, size=[400, 600])
	
end

# ╔═╡ 39aacdc0-7834-4ebd-9c09-73e2d1a01d09
begin

	fig2 = @layout [a b; c d; e f]

	
	p_lfp_early = plot(T, x_s, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=3.0,  lc=:red, xlims=(1,2), xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=13, xticks = 1:0.5:2)
	
	p_lfp_early = plot!(T, x_p, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=2.8, lc=:black, xlims=(1,2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=13, xticks = 1:0.25:2)
	
	title!("LFP: Earlier Learning")

	p_lfp_late = plot(T, x_s, xlabel= "Time (s)", ylabel="arb. V", label="Striatum", lw=3.0,  lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=13)
	
	p_lfp_late = plot!(T, x_p, xlabel= "Time (s)", ylabel="arb. V", label="Prefrontal Cortex", lw=2.8, lc=:black, xlims=(725,726.1), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=13, xticks = 725:0.5:726)
	
	title!("LFP: Later Learning")

	x_s_grab = Neuroblox.bandpassfilter(x_s, 15, 21, 1/dt, 6)
	x_s_angle = Neuroblox.phaseangle(x_s_grab)
	x_p_grab = Neuroblox.bandpassfilter(x_p, 15, 21, 1/dt, 6)
	x_p_angle = Neuroblox.phaseangle(x_p_grab)
	
	p_phase_early = plot(T, x_s_angle, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, xlims=(1,2), lw=2.0, grid = false, titlefontsize=13)
	
	p_phase_early = plot!(T, x_p_angle, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1,2), lw=2.0, grid = false, titlefontsize=13, ylabel="Angle")
	title!("Phase: Earlier Learning")

	p_phase_late = plot(T, x_s_angle, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, lw=2.0, grid = false, titlefontsize=13, ylabel="Angle")
	
	p_phase_late = plot!(T, x_p_angle, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, fg_legend = :false, legend = :outertop, lc=:black, xlims=(725,726.1), lw=2.0, grid = false, titlefontsize=13, xticks = 725:0.5:726)
	title!("Phase: Later Learning")

	f_se, power_se = Neuroblox.powerspectrum(x_s[Int(1):Int(floor(length(x_s)/10))], length(x_s[Int(1):Int(floor(length(x_s)/10))]), 1/dt, "pwelch", hanning)
	
	f_pe, power_pe = Neuroblox.powerspectrum(x_p[Int(1):Int(floor(length(x_p)/10))], length(x_p[Int(1):Int(floor(length(x_s)/10))]), 1/dt, "pwelch", hanning)

	p_psd_early = plot(f_se, power_se, label="Striatum", lw=3.0, lc=:red, xlabel="Frequency (Hz)", ylabel="arb. V/Hz^2", xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, titlefontsize=13)
	
	p_psd_early = plot!(f_pe, power_pe, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", ylabel="arb. V/Hz^2", lc=:black, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, titlefontsize=13)
	
	xlims!(0,50)
	title!("Power: Earlier Learning")

	f_sl, power_sl = Neuroblox.powerspectrum(x_s[Int(length(x_s)) - Int(floor(length(x_s)/10)):Int(length(x_s))], length(x_s[Int(length(x_s)) - Int(floor(length(x_s)/10)):Int(length(x_s))]), 1/dt, "pwelch", hanning)
	
	f_pl, power_pl = Neuroblox.powerspectrum(x_p[Int(length(x_p)) - Int(floor(length(x_p)/10)):Int(length(x_p))], length(x_p[Int(length(x_p)) - Int(floor(length(x_p)/10)):Int(length(x_p))]), 1/dt, "pwelch", hanning)

	p_psd_late = plot(f_sl, power_sl, label="Striatum", lw=3.0, lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=13)
	
	p_psd_late = plot!(f_pl, power_pl, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", ylabel="arb. V/Hz^2", lc=:black, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=13)
	xlims!(0,50)
	title!("Power: Later Learning")

	plot(p_lfp_early, p_lfp_late, p_phase_early, p_phase_late, p_psd_early, p_psd_late, layout = fig2, size=[680, 800])


end

# ╔═╡ Cell order:
# ╠═93e79dba-82a2-4574-a58f-cb1199fe254b
# ╠═8d65accf-023f-4e98-adaa-1a1584f770f1
# ╠═7b73249c-9254-4c3a-a5a4-20fd7ca310f2
# ╠═fd88ae6a-a3e6-46c4-a0d9-c8af6eb92dd5
# ╠═2d79cfd8-0ca1-418a-9065-6ac15764d972
# ╠═6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
# ╠═b7887fa0-bfd8-11ec-32cf-45269db719a2
# ╠═fddaed52-854d-4f5d-b2d4-f390ee74fa8a
# ╠═39aacdc0-7834-4ebd-9c09-73e2d1a01d09
