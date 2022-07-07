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
	phi = matread("phase_str.mat")
	phi_s1 = phi["phase_str"]
	phi = matread("phase_pfc.mat")
	phi_p1 = phi["phase_pfc"]
	phi = matread("phase_str_two.mat")
	phi_s2 = phi["phase_str_two"]
	phi = matread("phase_pfc_two.mat")
	phi_p2 = phi["phase_pfc_two"]
end

# ╔═╡ fd88ae6a-a3e6-46c4-a0d9-c8af6eb92dd5
begin
	tmin = 0.0
	dt = 0.001
	tmax = length(phi_p1)/2*dt
	T = tmin:dt:tmax
end

# ╔═╡ 2d79cfd8-0ca1-418a-9065-6ac15764d972
begin
	# Model Parameters
	
	a_s1 = 10
	a_p1 = 10
	ω_s1 = 45*(2*pi)
	k_s1 = ω_s1^2
	ω_p1 = 45*(2*pi)
	k_p1 = ω_p1^2
	
	a_s2 = 10
	a_p2 = 10
	ω_s2 = 45*(2*pi)
	k_s2 = ω_s2^2
	ω_p2 = 45*(2*pi)
	k_p2 = ω_p2^2
	
	h = 35.0
end

# ╔═╡ 6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
begin
	
	# Control Parameters

	Kp_s   = 0.05#0.1
	Kp_p   = 0.4#1.0
	Ki_s   = 0.05#0.1
	Ki_p   = 0.4#1.0
	
	g0   = rand(0.91:0.01:1.09, 8, 1)

	g_p1s1 = g0[1]*ones(length(T))
	g_s1p1 = g0[2]*ones(length(T)) 
	g_p2s2 = g0[3]*ones(length(T))
	g_s2p2 = g0[4]*ones(length(T)) 
	g_p2s1 = g0[5]
	g_s2p1 = g0[6]
	g_p1s2 = g0[7]
	g_s1p2 = g0[8]
	
	call_rate = Int(0.15/dt)
	controller_call_times = call_rate:call_rate:length(T)
	
end

# ╔═╡ cf2a24c4-244a-47f0-911c-ee6d6448bd6b
g0

# ╔═╡ 59ad4cfe-f3e2-4c3f-8d87-f46376c50545
begin
	
x_p1 = Array{Float64}(undef, length(T))
y_p1 = Array{Float64}(undef, length(T))
x_s1 = Array{Float64}(undef, length(T))
y_s1 = Array{Float64}(undef, length(T))
	
x_p2 = Array{Float64}(undef, length(T))
y_p2 = Array{Float64}(undef, length(T))
x_s2 = Array{Float64}(undef, length(T))
y_s2 = Array{Float64}(undef, length(T))

x_p1[1] = 0.1
y_p1[1] = 0.1
x_s1[1] = 0.1
y_s1[1] = 0.1

x_p2[1] = 0.1
y_p2[1] = 0.1
x_s2[1] = 0.1
y_s2[1] = 0.1
	
phi_x_p1 = []	
phi_x_s1 = []
circular_difference_11 = []

phi_x_p2 = []
phi_x_s2 = []	
circular_difference_22 = []

control_error_p1_s1 = []
control_error_s1_p1 = []
cumulative_error_p1_s1 = []
cumulative_error_s1_p1 = []
cumulative_error_all_11 = []

control_error_p2_s2 = []
control_error_s2_p2 = []
cumulative_error_s2_p2 = []
cumulative_error_p2_s2 = []
cumulative_error_all_22 = []

end

# ╔═╡ b7887fa0-bfd8-11ec-32cf-45269db719a2
begin

	for t = 1:length(T)-1
	
	    if t in controller_call_times[1:length(controller_call_times)-1]
			
			curr = Int(t/call_rate)
	
			x_p1_filt = Neuroblox.bandpassfilter(x_p1[(t+1-call_rate):t], 15, 21, 1/dt, 6)
	        push!(phi_x_p1, angle.(hilbert(x_p1_filt)))
			x_s1_filt = Neuroblox.bandpassfilter(x_s1[(t+1-call_rate):t], 15, 21, 1/dt, 6)
			push!(phi_x_s1, angle.(hilbert(x_s1_filt)))
	
			x_p2_filt = Neuroblox.bandpassfilter(x_p2[(t+1-call_rate):t], 15, 21, 1/dt, 6)
	        push!(phi_x_p2, angle.(hilbert(x_p2_filt)))
			x_s2_filt = Neuroblox.bandpassfilter(x_s2[(t+1-call_rate):t], 15, 21, 1/dt, 6)
			push!(phi_x_s2, angle.(hilbert(x_s2_filt)))
			
			diff_11 = exp.(im.*phi_x_p1[curr]) ./ exp.(im.*phi_x_s1[curr])
	        push!(circular_difference_11, angle.(diff_11))
			
			diff_22 = exp.(im.*phi_x_p2[curr]) ./ exp.(im.*phi_x_s2[curr])
	        push!(circular_difference_22, angle.(diff_22))
	
			if mean(circular_difference_11[curr]) > 0
				push!(control_error_p1_s1, mean(circular_difference_11[curr]))
				push!(cumulative_error_p1_s1, cumsum(control_error_p1_s1))
				push!(control_error_s1_p1, 0)
				push!(cumulative_error_s1_p1, cumsum(control_error_s1_p1))
			end
			if mean(circular_difference_11[curr]) < 0
				push!(control_error_s1_p1, mean(circular_difference_11[curr]))
				push!(cumulative_error_s1_p1, cumsum(control_error_s1_p1))	
				push!(control_error_p1_s1, 0)
				push!(cumulative_error_p1_s1, cumsum(control_error_p1_s1))
			end
			cumulative_error_11 = last(cumulative_error_p1_s1[length(cumulative_error_p1_s1)]) + last(cumulative_error_s1_p1[length(cumulative_error_s1_p1)]) 
			push!(cumulative_error_all_11, cumulative_error_11)
	
			if mean(circular_difference_11[curr]) > 0
				g_p1s1[t:t+call_rate] .= g0[1] .+ Kp_p.*(abs(control_error_p1_s1[curr])) .+ Ki_p.*(cumulative_error_11)
				g_s1p1[t:t+call_rate] .= g_s1p1[t-1]
			end
			if mean(circular_difference_11[curr]) < 0
				g_p1s1[t:t+call_rate] .= g_p1s1[t-1] 
				g_s1p1[t:t+call_rate] .= g0[2] .+ Kp_s.*(abs(control_error_s1_p1[curr])) .+ Ki_s.*(cumulative_error_11)
			end
			if mean(circular_difference_11[curr]) == 0
				g_p1s1[t:t+call_rate] .= g_p1s1[t-1]
				g_s1p1[t:t+call_rate] .= g_s1p1[t-1]
			end
	
			if mean(circular_difference_22[curr]) > 0
				push!(control_error_p2_s2, mean(circular_difference_22[curr]))
				push!(cumulative_error_p2_s2, cumsum(control_error_p2_s2))
				push!(control_error_s2_p2, 0)
				push!(cumulative_error_s2_p2, cumsum(control_error_s2_p2))
			end
			if mean(circular_difference_22[curr]) < 0
				push!(control_error_s2_p2, mean(circular_difference_22[curr]))
				push!(cumulative_error_s2_p2, cumsum(control_error_s2_p2))	
				push!(control_error_p2_s2, 0)
				push!(cumulative_error_p2_s2, cumsum(control_error_p2_s2))
			end
			cumulative_error_22 = last(cumulative_error_p2_s2[length(cumulative_error_p2_s2)]) + last(cumulative_error_s2_p2[length(cumulative_error_s2_p2)])
			push!(cumulative_error_all_22, cumulative_error_22)
	
	
			if mean(circular_difference_22[curr]) > 0
				g_p2s2[t:t+call_rate] .= g0[3] .+ Kp_p.*(abs(control_error_p2_s2[curr])) .+ Ki_p.*(cumulative_error_22)
				g_s2p2[t:t+call_rate] .= g_s2p2[t-1]
			end
			if mean(circular_difference_22[curr]) < 0
				g_p2s2[t:t+call_rate] .= g_p2s2[t-1] 
				g_s2p2[t:t+call_rate] .= g0[4] .+ Kp_s.*(abs(control_error_s2_p2[curr])) .+ Ki_s.*(cumulative_error_22)
			end
			if mean(circular_difference_22[curr]) == 0
				g_p2s2[t:t+call_rate] .= g_p2s2[t-1]
				g_s2p2[t:t+call_rate] .= g_s2p2[t-1]
			end
	
	    end
	
	    dx_s1 = y_s1[t]-(2ω_s1*x_s1[t])+ k_s1*(2/π)*atan(g_p1s1[t]*(x_p1[t]/h) + a_s1*cos(ω_s1*t + phi_s1[t]) + g0[5]*(x_p2[t]/h))
		
	    dy_s1 = -(ω_s1^2)*x_s1[t]	
		
	    dx_p1 = y_p1[t]-(2*ω_p1*x_p1[t])+ k_p1*(2/π)*atan(g_s1p1[t]*(x_s1[t]/h) + a_p1*cos(ω_p1*t + phi_p1[t]) + g0[6]*(x_s2[t]/h))
		
	    dy_p1 = -(ω_p1^2)*x_p1[t]
	
		dx_s2 = y_s2[t]-(2ω_s2*x_s2[t])+ k_s2*(2/π)*atan(g_p2s2[t]*(x_p2[t]/h) + a_s2*cos(ω_s2*t + phi_s2[t]) + g0[7]*(x_p1[t]/h))
		
	    dy_s2 = -(ω_s2^2)*x_s2[t]	
		
	    dx_p2 = y_p2[t]-(2*ω_p2*x_p2[t])+ k_p2*(2/π)*atan(g_s2p2[t]*(x_s2[t]/h) + a_p2*cos(ω_p2*t + phi_p2[t]) + g0[8]*(x_s1[t]/h))
		
	    dy_p2 = -(ω_p2^2)*x_p2[t]
		
		#RK4 Application
	
		k1_xs = dt*dx_s1
		k2_xs = (dt/2)*(dx_s1 + k1_xs/2)
		k3_xs = (dt/2)*(dx_s1 + k2_xs/2)
		k4_xs = (dt/2)*(dx_s1 + k3_xs/2)
	    x_s1[t+1] = x_s1[t] + (k1_xs + 2*k2_xs + 2*k3_xs + k4_xs)/6
			
		k1_ys = dt*dy_s1
		k2_ys = (dt/2)*(dy_s1 + k1_ys/2)
		k3_ys = (dt/2)*(dy_s1 + k2_ys/2)
		k4_ys = (dt/2)*(dy_s1 + k3_ys/2)
	    y_s1[t+1] = y_s1[t] + (k1_ys + 2*k2_ys + 2*k3_ys + k4_ys)/6
			
		k1_xp = dt*dx_p1
		k2_xp = (dt/2)*(dx_p1 + k1_xp/2)
		k3_xp = (dt/2)*(dx_p1 + k2_xp/2)
		k4_xp = (dt/2)*(dx_p1 + k3_xp/2)
	    x_p1[t+1] = x_p1[t] + (k1_xp + 2*k2_xp + 2*k3_xp + k4_xp)/6
			
		k1_yp = dt*dy_p1
		k2_yp = (dt/2)*(dy_p1 + k1_yp/2)
		k3_yp = (dt/2)*(dy_p1 + k2_yp/2)
		k4_yp = (dt/2)*(dy_p1 + k3_yp/2)
	    y_p1[t+1] = y_p1[t] + (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)/6
		
		k1_xs = dt*dx_s2
		k2_xs = (dt/2)*(dx_s2 + k1_xs/2)
		k3_xs = (dt/2)*(dx_s2 + k2_xs/2)
		k4_xs = (dt/2)*(dx_s2 + k3_xs/2)
	    x_s2[t+1] = x_s2[t] + (k1_xs + 2*k2_xs + 2*k3_xs + k4_xs)/6
			
		k1_ys = dt*dy_s2
		k2_ys = (dt/2)*(dy_s2 + k1_ys/2)
		k3_ys = (dt/2)*(dy_s2 + k2_ys/2)
		k4_ys = (dt/2)*(dy_s2 + k3_ys/2)
	    y_s2[t+1] = y_s2[t] + (k1_ys + 2*k2_ys + 2*k3_ys + k4_ys)/6
			
		k1_xp = dt*dx_p2
		k2_xp = (dt/2)*(dx_p2 + k1_xp/2)
		k3_xp = (dt/2)*(dx_p2 + k2_xp/2)
		k4_xp = (dt/2)*(dx_p2 + k3_xp/2)
	    x_p2[t+1] = x_p2[t] + (k1_xp + 2*k2_xp + 2*k3_xp + k4_xp)/6
			
		k1_yp = dt*dy_p2
		k2_yp = (dt/2)*(dy_p2 + k1_yp/2)
		k3_yp = (dt/2)*(dy_p2 + k2_yp/2)
		k4_yp = (dt/2)*(dy_p2 + k3_yp/2)
	    y_p2[t+1] = y_p2[t] + (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)/6
				
	end
end

# ╔═╡ fddaed52-854d-4f5d-b2d4-f390ee74fa8a
begin

	fig1 = @layout [a b; c d; e f]
	
	p_ce11 = plot(control_error_s1_p1[1:length(controller_call_times)-1], label="e(S1->P1)", linecolor = "red", lw=2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100/2), ylims=(-3.14,3.14))
	p_ce11 = plot!(control_error_p1_s1[1:length(controller_call_times)-1], label="e(P1->S1)", linecolor ="black", lw=:2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100/2), grid = false, ylims=(-3.14,3.14), xticks = 0:750:4500, titlefontsize=11)
	title!("Control Error (S1<->P1)")

	p_cumul11 = plot(cumulative_error_all_11[1:length(controller_call_times)-1], linecolor ="gold", lw=:2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlabel = "Controller Call #", ylabel="cumule(t)", xlims=(0,5100/2), grid = false, xticks = 0:750:4500, titlefontsize=11, label=false)
	title!("Cumulative Error (S1<->P1)")
	
	p_gain11 = plot(T, g_s1p1, label="g(S1->P1)", lw=2.5, lc=:red, fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9)
	p_gain11 = plot!(T, g_p1s1, label="g(P1->S1)", lw=2.3, lc=:black, fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlims=(0, tmax-1), grid = false, xticks = 0:75:725, titlefontsize=11)
	title!("Reinforce Plasticity (S1<->P1)")
	xlabel!("Time (s)")
	ylabel!("Edge Weights")

	p_ce22 = plot(control_error_s2_p2[1:length(controller_call_times)-1], label="e(S2->P2)", linecolor = "red", lw=2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100), ylims=(-3.14,3.14))
	p_ce22 = plot!(control_error_p2_s2[1:length(controller_call_times)-1], label="e(P2->S2)", linecolor ="black", lw=:2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlabel = "Controller Call #", ylabel="e(t)", xlims=(0,5100/2), grid = false, ylims=(-3.14,3.14), xticks = 0:750:4500, titlefontsize=11)
	title!("Control Error (S2<->P2)")

	p_cumul22 = plot(cumulative_error_all_22[1:length(controller_call_times)-1], linecolor ="gold", lw=:2.0, fg_legend = :false, legend = :outertop, tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlabel = "Controller Call #", ylabel="cumule(t)", xlims=(0,5100/2), grid = false, xticks = 0:750:4500, titlefontsize=11,label=false)
	title!("Cumulative Error (S2<->P2)")
	
	p_gain22 = plot(T, g_s2p2, label="g(S2->P2)", lw=2.5, lc=:red, fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9)
	p_gain22 = plot!(T, g_p2s2, label="g(P2->S2)", lw=2.3, lc=:black, fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlims=(0, tmax-1), grid = false, xticks = 0:75:725, titlefontsize=11)
	title!("Reinforce Plasticity (S2<->P2)")
	xlabel!("Time (s)")
	ylabel!("Edge Weights")

	plot(p_ce11, p_ce22, p_cumul11, p_cumul22, p_gain11, p_gain22, layout = fig1, size=[800, 700])
	
end

# ╔═╡ 39aacdc0-7834-4ebd-9c09-73e2d1a01d09
begin

	fig2 = @layout [a b; c d]

	p_lfp_early11 = plot(T, x_s1, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xlims=(1,2), xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.5:2)
	
	p_lfp_early11 = plot!(T, x_p1, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(1,2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=11, xticks = 1:0.25:2, ylabel="arb. V")
	title!("LFP: Earlier Learning (S1 <-> P1)")

	p_lfp_early22 = plot(T, x_s2, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xlims=(1,2), xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.5:2)
	
	p_lfp_early22 = plot!(T, x_p2, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(1,2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.25:2, ylabel="arb. V")
	title!("LFP: Earlier Learning (S2 <-> P2)")

	p_lfp_late11 = plot(T, x_s1, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=11)
	
	p_lfp_late11 = plot!(T, x_p1, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(725/2,726.1/2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 725/2:0.5:726/2, ylabel="arb. V")
	title!("LFP: Later Learning (S1 <-> P1)")

	p_lfp_late22 = plot(T, x_s2, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11)
	
	p_lfp_late22 = plot!(T, x_p2, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(725/2,726.1/2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 725/2:0.5:726/2, ylabel="arb. V")
	title!("LFP: Later Learning (S2 <-> P2)")


	plot(p_lfp_early11, p_lfp_late11, p_lfp_early22, p_lfp_late22, layout = fig2, size=[650, 400])


end

# ╔═╡ 8c81c812-a3ba-4d6b-8157-6a90ea983b66
begin

	fig2b = @layout [a b; c d]

	p_lfp_early12 = plot(T, x_s1, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xlims=(1,2), xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.5:2)
	
	p_lfp_early12 = plot!(T, x_p2, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(1,2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=11, xticks = 1:0.25:2, ylabel="arb. V")
	title!("LFP: Earlier Learning (S1 <-> P2)")

	p_lfp_early21 = plot(T, x_s2, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xlims=(1,2), xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.5:2)
	
	p_lfp_early21 = plot!(T, x_p1, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(1,2), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 1:0.25:2, ylabel="arb. V")
	title!("LFP: Earlier Learning (S2 <-> P1)")

	p_lfp_late12 = plot(T, x_s1, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, grid = false, titlefontsize=11)
	
	p_lfp_late12 = plot!(T, x_p2, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(725,726.1), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="arb. V")
	title!("LFP: Later Learning (S1 <-> P2)")

	p_lfp_late21 = plot(T, x_s2, xlabel= "Time (s)", label="Striatum", lw=2.1,  lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11)
	
	p_lfp_late21 = plot!(T, x_p1, xlabel= "Time (s)", label="Prefrontal Cortex", lw=1.8, lc=:black, xlims=(725,726.1), fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="arb. V")
	title!("LFP: Later Learning (S2 <-> P1)")


	plot(p_lfp_early12, p_lfp_late12, p_lfp_early21, p_lfp_late21, layout = fig2b, size=[650, 400])


end

# ╔═╡ e9e6a7b6-16c2-4d3e-928b-ba7fe48f8464
begin

	fig22 = @layout [e f; g h; i j; k l]

	x_s_grab = Neuroblox.bandpassfilter(x_s1, 15, 21, 1/dt, 6)
	x_s_angle = Neuroblox.phaseangle(x_s_grab)
	x_p_grab = Neuroblox.bandpassfilter(x_p1, 15, 21, 1/dt, 6)
	x_p_angle = Neuroblox.phaseangle(x_p_grab)
	
	p_phase_early11 = plot(T, x_s_angle, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, xlims=(1,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)")
	p_phase_early11 = plot!(T, x_p_angle, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1.5,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Earlier Learning (S1 <-> P1)")

	p_phase_late11 = plot(T, x_s_angle, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, lc=:red, lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)")
	p_phase_late11 = plot!(T, x_p_angle, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(725.5,726.1), lw=2.0, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Later Learning (S1 <-> P1)")

	x_s_grab = Neuroblox.bandpassfilter(x_s2, 15, 21, 1/dt, 6)
	x_s_angle2 = Neuroblox.phaseangle(x_s_grab)
	x_p_grab = Neuroblox.bandpassfilter(x_p2, 15, 21, 1/dt, 6)
	x_p_angle2 = Neuroblox.phaseangle(x_p_grab)
	
	p_phase_early22 = plot(T, x_s_angle2, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, lc=:red, xlims=(1,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	p_phase_early22 = plot!(T, x_p_angle2, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1.5,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Earlier Learning (S2 <-> P2)")

	p_phase_late22 = plot(T, x_s_angle2, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	p_phase_late22 = plot!(T, x_p_angle2, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(725.5,726.1), lw=2.0, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Later Learning (S2 <-> P2)")

	f_se, power_se = Neuroblox.powerspectrum(x_s1[Int(1):Int(floor(length(x_s1)/10))], length(x_s1[Int(1):Int(floor(length(x_s1)/10))]), 1/dt, "pwelch", hanning)
	f_pe, power_pe = Neuroblox.powerspectrum(x_p1[Int(1):Int(floor(length(x_p1)/10))], length(x_p1[Int(1):Int(floor(length(x_p1)/10))]), 1/dt, "pwelch", hanning)

	p_psd_early11 = plot(f_se, power_se, label="Striatum", lw=3.0, lc=:red, xlabel="Frequency (Hz)", xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, legend = :outertop, yguidefontsize=12,legendfontsize=12, fg_legend = :false, grid = false, titlefontsize=15, ylabel="arb. V^2/Hz")
	p_psd_early11 = plot!(f_pe, power_pe, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(0,50)
	title!("Power: Earlier Learning (S1 <-> P1)")

	f_sl, power_sl = Neuroblox.powerspectrum(x_s1[Int(length(x_s1)) - Int(floor(length(x_s1)/10)):Int(length(x_s1))], length(x_s1[Int(length(x_s1)) - Int(floor(length(x_s1)/10)):Int(length(x_s1))]), 1/dt, "pwelch", hanning)
	f_pl, power_pl = Neuroblox.powerspectrum(x_p1[Int(length(x_p1)) - Int(floor(length(x_p1)/10)):Int(length(x_p1))], length(x_p1[Int(length(x_p1)) - Int(floor(length(x_p1)/10)):Int(length(x_p1))]), 1/dt, "pwelch", hanning)

	p_psd_late11 = plot(f_sl, power_sl, label="Striatum", lw=3.0, lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_late11 = plot!(f_pl, power_pl, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(10,30)
	title!("Power: Later Learning (S1 <-> P1)")

	f_se, power_se = Neuroblox.powerspectrum(x_s2[Int(1):Int(floor(length(x_s2)/10))], length(x_s2[Int(1):Int(floor(length(x_s2)/10))]), 1/dt, "pwelch", hanning)
	f_pe, power_pe = Neuroblox.powerspectrum(x_p2[Int(1):Int(floor(length(x_p2)/10))], length(x_p2[Int(1):Int(floor(length(x_p2)/10))]), 1/dt, "pwelch", hanning)

	p_psd_early22 = plot(f_se, power_se, label="Striatum", lw=3.0, lc=:red, xlabel="Frequency (Hz)", xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_early22 = plot!(f_pe, power_pe, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(0,50)
	title!("Power: Earlier Learning (S2 <-> P2)")

	f_sl, power_sl = Neuroblox.powerspectrum(x_s2[Int(length(x_s2)) - Int(floor(length(x_s2)/10)):Int(length(x_s2))], length(x_s2[Int(length(x_s2)) - Int(floor(length(x_s2)/10)):Int(length(x_s2))]), 1/dt, "pwelch", hanning)
	f_pl, power_pl = Neuroblox.powerspectrum(x_p2[Int(length(x_p2)) - Int(floor(length(x_p2)/10)):Int(length(x_p2))], length(x_p2[Int(length(x_p2)) - Int(floor(length(x_p2)/10)):Int(length(x_p2))]), 1/dt, "pwelch", hanning)

	p_psd_late22 = plot(f_sl, power_sl, label="Striatum", lw=3.0, lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_late22 = plot!(f_pl, power_pl, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(10,30)
	title!("Power: Later Learning (S2 <-> P2)")

	plot(p_phase_early11, p_phase_late11, p_phase_early22, p_phase_late22, p_psd_early11, p_psd_late11, p_psd_early22, p_psd_late22, layout = fig22, size=[900, 900])


end

# ╔═╡ a2dc1d77-171c-4428-a1aa-bb670be14151
begin

	fig22b = @layout [e f; g h; i j; k l]

	x_s_grabb = Neuroblox.bandpassfilter(x_s1, 15, 21, 1/dt, 6)
	x_s_angleb = Neuroblox.phaseangle(x_s_grabb)
	x_p_grabb = Neuroblox.bandpassfilter(x_p2, 15, 21, 1/dt, 6)
	x_p_angleb = Neuroblox.phaseangle(x_p_grabb)
	
	p_phase_early12 = plot(T, x_s_angleb, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, xlims=(1,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)")
	p_phase_early12 = plot!(T, x_p_angleb, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=10, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1.5,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Earlier Learning (S1 <-> P2)")

	p_phase_late12 = plot(T, x_s_angleb, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, lc=:red, lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)")
	p_phase_late12 = plot!(T, x_p_angleb, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(725.5,726.1), lw=2.0, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Later Learning (S1 <-> P2)")

	x_s_grabb = Neuroblox.bandpassfilter(x_s2, 15, 21, 1/dt, 6)
	x_s_angle2b = Neuroblox.phaseangle(x_s_grabb)
	x_p_grabb = Neuroblox.bandpassfilter(x_p1, 15, 21, 1/dt, 6)
	x_p_angle2b = Neuroblox.phaseangle(x_p_grabb)
	
	p_phase_early21 = plot(T, x_s_angle2b, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, lc=:red, xlims=(1,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	p_phase_early21 = plot!(T, x_p_angle2b, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(1.5,2), lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Earlier Learning (S2 <-> P1)")

	p_phase_late21 = plot(T, x_s_angle2b, label="phi_S", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, lc=:red, lw=2.0, grid = false, titlefontsize=11, ylabel="Angle (rad)", xlabel="Time (s)")
	p_phase_late21 = plot!(T, x_p_angle2b, label = "phi_P", tickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, legend = :outertop, lc=:black, xlims=(725.5,726.1), lw=2.0, grid = false, titlefontsize=11, xticks = 725:0.5:726, ylabel="Angle (rad)", xlabel="Time (s)")
	title!("Phase: Later Learning (S2 <-> P1)")

	f_seb, power_seb = Neuroblox.powerspectrum(x_s1[Int(1):Int(floor(length(x_s1)/10))], length(x_s1[Int(1):Int(floor(length(x_s1[:,1])/10))]), 1/dt, "pwelch", hanning)
	f_peb, power_peb = Neuroblox.powerspectrum(x_p2[Int(1):Int(floor(length(x_p2)/10))], length(x_p2[Int(1):Int(floor(length(x_p2[:,1])/10))]), 1/dt, "pwelch", hanning)

	p_psd_early12 = plot(f_seb, power_seb, label="Striatum", lw=3.0, lc=:red, xlabel="Frequency (Hz)", xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, legend = :outertop, yguidefontsize=12,legendfontsize=12, fg_legend = :false, grid = false, titlefontsize=15, ylabel="arb. V^2/Hz")
	p_psd_early12 = plot!(f_peb, power_peb, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(0,50)
	title!("Power: Earlier Learning (S1 <-> P2)")

	f_slb, power_slb = Neuroblox.powerspectrum(x_s1[Int(length(x_s1)) - Int(floor(length(x_s1)/10)):Int(length(x_s1))], length(x_s1[Int(length(x_s1)) - Int(floor(length(x_s1)/10)):Int(length(x_s1)),1]), 1/dt, "pwelch", hanning)
	f_plb, power_plb = Neuroblox.powerspectrum(x_p2[Int(length(x_p2)) - Int(floor(length(x_p2)/10)):Int(length(x_p2))], length(x_p2[Int(length(x_p2)) - Int(floor(length(x_p2)/10)):Int(length(x_p2)),1]), 1/dt, "pwelch", hanning)

	p_psd_late12 = plot(f_slb, power_slb, label="Striatum", lw=3.0, lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_late12 = plot!(f_plb, power_plb, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(10,30)
	title!("Power: Later Learning (S1 <-> P2)")

	f_seb, power_seb = Neuroblox.powerspectrum(x_s2[Int(1):Int(floor(length(x_s2)/10))], length(x_s2[Int(1):Int(floor(length(x_s2)/10))]), 1/dt, "pwelch", hanning)
	f_peb, power_peb = Neuroblox.powerspectrum(x_p1[Int(1):Int(floor(length(x_p1)/10))], length(x_p1[Int(1):Int(floor(length(x_p1)/10))]), 1/dt, "pwelch", hanning)

	p_psd_early21 = plot(f_seb, power_seb, label="Striatum", lw=3.0, lc=:red, xlabel="Frequency (Hz)", xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=10, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_early21 = plot!(f_peb, power_peb, label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(0,50)
	title!("Power: Earlier Learning (S2 <-> P1)")

	f_slb, power_slb = Neuroblox.powerspectrum(x_s2[Int(length(x_s2)) - Int(floor(length(x_s2)/10)):Int(length(x_s2))], length(x_s2[Int(length(x_s2)) - Int(floor(length(x_s2)/10)):Int(length(x_s2))]), 1/dt, "pwelch", hanning)
	f_plb, power_plb = Neuroblox.powerspectrum(x_p1[Int(length(x_p1)) - Int(floor(length(x_p1)/10)):Int(length(x_p1))], length(x_p1[Int(length(x_p1)) - Int(floor(length(x_p1)/10)):Int(length(x_p1))]), 1/dt, "pwelch", hanning)

	p_psd_late21 = plot(f_slb, power_slb, label="Striatum", lw=3.0, lc=:red, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, legend = :outertop, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	p_psd_late21 = plot!(f_plb, power_plb , label="Prefrontal Cortex", lw=3.0, xlabel="Frequency (Hz)", lc=:black, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, fg_legend = :false, grid = false, xticks = 0:5:50, titlefontsize=11, ylabel="arb. V^2/Hz")
	xlims!(10,30)
	title!("Power: Later Learning (S2 <-> P1)")

	plot(p_phase_early12, p_phase_late12, p_phase_early21, p_phase_late21, p_psd_early12, p_psd_late12, p_psd_early21, p_psd_late21, layout = fig22b, size=[900, 900])


end

# ╔═╡ 59a826f5-e240-4a7d-9ca7-c57d47521285
begin
	fig3 = @layout [a;b]
	
	aa = plot(T, g_p1s1, label="g(P1->S1)", lw=2.3, lc=:black, fg_legend = :false, legend = :outertop, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlims=(0, tmax-1), grid = false, xticks = 0:75:725, titlefontsize=11)
	aa = plot!(T, g_s1p1, label="g(S1->P1)", lw=2.5, lc=:red, fg_legend = :false, legend = :bottomleft, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9)
	title!("Reinforce Plasticity (P1<->S1)")
	xlabel!("Time (s)")
	ylabel!("Edge Weights")
	aa = plot!(T, (g_s1p1 .+ g_p1s1)/2, lw=2.8, lc=:purple, fg_legend=:false, label="g(P1<->S1)")

	bb = plot(T, g_p2s2, label="g(P2->S2)", lw=2.3, lc=:black, fg_legend = :false, legend = :bottomleft, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9, xlims=(0, tmax-1), grid = false, xticks = 0:75:725, titlefontsize=11)
	bb = plot!(T, g_s2p2, label="g(S2->P2)", lw=2.5, lc=:red, fg_legend = :false, legend = :topleft, xtickfontsize=10,ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,legendfontsize=9)
	title!("Reinforce Plasticity (P2 <-> S2)")
	xlabel!("Time (s)")
	ylabel!("Edge Weights")
	bb = plot!(T, (g_s2p2 .+ g_p2s2)/2, lw=2.8, lc=:gold, fg_legend=:false, label="g(P2<->S2)")

	plot(aa, bb, layout = fig3, size=[410,450])
	
end

# ╔═╡ 349569a8-09d3-4c4a-b5a7-b9354cd94878
begin
plot(T, (g_s1p1 .+ g_p1s1)/2, lw=3.8, lc=:purple, fg_legend=:false, label="Category A")
plot!(T, (g_s2p2 .+ g_p2s2)/2, lw=3.8, lc=:gold, fg_legend=:false, label="Category B", xlims=(0, tmax-1), grid = false, xticks = 0:75:725, legend = :outertop, ylabel="Edge Weight Average", xlabel="Time (s)", title="Learning Weights", xtickfontsize=12,ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,legendfontsize=12,titlefontsize=17)
end

# ╔═╡ Cell order:
# ╠═93e79dba-82a2-4574-a58f-cb1199fe254b
# ╠═8d65accf-023f-4e98-adaa-1a1584f770f1
# ╠═7b73249c-9254-4c3a-a5a4-20fd7ca310f2
# ╠═fd88ae6a-a3e6-46c4-a0d9-c8af6eb92dd5
# ╠═2d79cfd8-0ca1-418a-9065-6ac15764d972
# ╠═6ca0b6c9-db42-42f6-97d8-f2427f5ab15b
# ╠═cf2a24c4-244a-47f0-911c-ee6d6448bd6b
# ╠═59ad4cfe-f3e2-4c3f-8d87-f46376c50545
# ╠═b7887fa0-bfd8-11ec-32cf-45269db719a2
# ╠═fddaed52-854d-4f5d-b2d4-f390ee74fa8a
# ╠═39aacdc0-7834-4ebd-9c09-73e2d1a01d09
# ╠═8c81c812-a3ba-4d6b-8157-6a90ea983b66
# ╠═e9e6a7b6-16c2-4d3e-928b-ba7fe48f8464
# ╠═a2dc1d77-171c-4428-a1aa-bb670be14151
# ╠═59a826f5-e240-4a7d-9ca7-c57d47521285
# ╠═349569a8-09d3-4c4a-b5a7-b9354cd94878
