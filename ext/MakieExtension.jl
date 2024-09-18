module MakieExtension

isdefined(Base, :get_extension) ? using Makie : using ..Makie

using Neuroblox
using Neuroblox: AbstractNeuronBlox, CompositeBlox, VLState, VLSetup
using Neuroblox: meanfield_timeseries, voltage_timeseries, detect_spikes, get_neurons
using SciMLBase: AbstractSolution
using LinearAlgebra: diag

import Neuroblox: meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, voltage_stack, effectiveconnectivity, effectiveconnectivity!, ecbarplot, freeenergy, freeenergy!


@recipe(FreeEnergy, spDCMresults) do scene
    Theme()
end

argument_names(::Type{<: FreeEnergy}) = (:spDCMresults)

function Makie.plot!(p::FreeEnergy)
    F = p.spDCMresults[].F
    deleteat!(F, 1)   # remove the first value since that's always -Inf
    
    lines!(p, 1:length(F), F)
    scatter!(p, 1:length(F), F)
    return p
end

@recipe(EffectiveConnectivity, spDCMresults, spDCMsetup, groundtruth) do scene
    Theme()
end

argument_names(::Type{<: EffectiveConnectivity}) = (:spDCMresults, :spDCMsetup, :groundtruth)

function Makie.plot!(p::EffectiveConnectivity)
    nr = p.spDCMsetup[].systemnums[1]  # number of regions
    diagidx = 1:(nr+1):nr^2
    gt = vec(p.groundtruth[])   # get ground truth values
    deleteat!(gt, diagidx)
    state = p.spDCMresults[]
    μA = state.μθ_po[1:nr^2]    # get estimated means of effective connectivity
    deleteat!(μA, diagidx)
    var_A = diag(state.Σθ_po[1:nr^2, 1:nr^2])  # get variance of effective connectivity
    deleteat!(var_A, diagidx)
    x = 1:(nr^2-nr)
    barplot!(p, x, μA)
    errorbars!(p, x, μA, sqrt.(var_A), color = :red)
    scatter!(p, x, gt)
    return p
end

function ecbarplot(spDCMresults::VLState, spDCMsetup::VLSetup, groundtruth)
    nr = spDCMsetup.systemnums[1]  # number of regions
    modelparam = spDCMsetup.modelparam
    fig = Figure()
    idx = collect(1:nr^2)
    deleteat!(idx, 1:(nr+1):nr^2)
    xlabels = string.(collect(keys(modelparam))[idx])
    ax = Axis(fig[1,1], xticks = (1:(nr^2-nr), xlabels))

    effectiveconnectivity!(ax, spDCMresults, spDCMsetup, groundtruth)
    display(fig)
end

@recipe(MeanField, blox, sol) do scene
    Theme()
end

argument_names(::Type{<: MeanField}) = (:blox, :sol)

function Makie.plot!(p::MeanField)
    sol = p.sol[]
    blox = p.blox[]

    V = meanfield_timeseries(blox, sol)
    
    lines!(p, sol.t, vec(V))

    return p
end

@recipe(RasterPlot, blox, sol) do scene
    Theme(
        color = :black
    )
end

argument_names(::Type{<: RasterPlot}) = (:blox, :sol)

function Makie.plot!(p::RasterPlot)
    sol = p.sol[]
    t = sol.t
    blox = p.blox[]
    neurons = get_neurons(blox)
    
    for (i, n) in enumerate(neurons)
        spike_idxs = detect_spikes(n, sol)
        scatter!(p, t[spike_idxs], fill(i, length(spike_idxs)); color=p.color[])
    end
    
    return p
end

@recipe(StackPlot, blox, sol) do scene
    Theme(
        color = :black
    )
end

argument_names(::Type{<: StackPlot}) = (:blox, :sol)

function Makie.plot!(p::StackPlot)
    sol = p.sol[]
    blox = p.blox[]

    V = voltage_timeseries(blox, sol)
    
    offset = 20
    for (i,V_neuron) in enumerate(eachcol(V))
        lines!(p, sol.t, (i-1)*offset .+ V_neuron; color=p.color[])
    end
    
    return p
end

function Makie.convert_arguments(::Makie.PointBased, blox::AbstractNeuronBlox, sol::AbstractSolution)
    V = voltage_timeseries(blox, sol)

    return (sol.t, V)
end

function voltage_stack(blox::CompositeBlox, sol::AbstractSolution; N_neurons=10, fontsize=8, color=:black)
    neurons = get_neurons(blox)
    N_ax = min(length(neurons), N_neurons)

    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Time", ylabel="Neurons")

    hideydecorations!(ax)

    stackplot!(ax, blox, sol)

    display(fig)
end

@recipe(PowerSpectrum, pergram) do scene
    Theme()
end

argument_names(::Type{<: PowerSpectrum}) = (:pergram)

function Makie.plot!(p::PowerSpectrum)

    powspec = p.pergram[]
    
    lines!(p, powspec.freq[2:end], powspec.power[2:end])

    return p
end

function power_composite(blox::CompositeBlox, sol::AbstractSolution)
    pergram = meanfield_powerspectrum(blox, sol)

    fig = Figure()
    ax = Axis(fig[1,1],
             xlabel="f in Hz",
             ylabel="Power Spectrum",
             xticks = [8,12,20,30, 40, 50,60,70,80,90])

    xlims!(ax, 8, 100)
    ylims!(ax, 0, 0.2)
    powerspectrum!(ax, pergram)

    poly!(ax, Point2f[(8, 0), (8, 1), (12, 1), (12, 0)], color = (:red,0.2), strokecolor = :black, strokewidth = 1)
	
	poly!(ax, Point2f[(12, 0), (12, 1), (35, 1), (35, 0)], color = (:blue,0.2), strokecolor = :black, strokewidth = 1)
	
	poly!(ax, Point2f[(35, 0), (35, 1), (200, 1), (200, 0)], color = (:green,0.2), strokecolor = :black, strokewidth = 1)
	
	text!(ax, 8.5, 0.17; text=L"\alpha", fontsize=24)
	text!(ax, 22, 0.17; text=L"\beta", fontsize=24)
	text!(ax, 60, 0.17; text=L"\gamma", fontsize=24)
	
	axislegend(ax, position=:rt)
    display(fig)
end

end