module MakieExtension

isdefined(Base, :get_extension) ? using Makie : using ..Makie

using Neuroblox
using Neuroblox: AbstractBlox, AbstractNeuronBlox, CompositeBlox, VLState, VLSetup
using Neuroblox: meanfield_timeseries, voltage_timeseries, detect_spikes, firing_rate, get_neurons
using Neuroblox: powerspectrum
using SciMLBase: AbstractSolution, EnsembleSolution
using LinearAlgebra: diag
using SparseArrays
using DSP
using Statistics: mean, std


import Neuroblox: meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, frplot, frplot!, voltage_stack, effectiveconnectivity, effectiveconnectivity!, ecbarplot, freeenergy, freeenergy!
import Neuroblox: powerspectrumplot, powerspectrumplot!

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
        color = :black,
        threshold = nothing,
        title = "",
        Axis = (
            xlabel = "Time (ms)",
            ylabel = "Neurons"
        )
    )
end

argument_names(::Type{<: RasterPlot}) = (:blox, :sol)

function Makie.plot!(p::RasterPlot)
    sol = p.sol[]
    t = sol.t
    blox = p.blox[]
    threshold = p.threshold[]

    ax = current_axis()
    ax.xlabel = p.Axis.xlabel[]
    ax.ylabel = p.Axis.ylabel[]
    ax.title = p.title[]

    spikes = detect_spikes(blox, sol; threshold=threshold)
    spike_times, neuron_indices = findnz(spikes)
    scatter!(p, sol.t[spike_times], neuron_indices; color=p.color[])

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
    
    V = V .- mean(V; dims = 1)

    mx = maximum(V; dims = 1)
    mn = minimum(V; dims = 1)
    
    offset = 0.0
    for (i, V_neuron) in enumerate(eachcol(V))
        if i == 1
            lines!(p, sol.t, V_neuron; color=p.color[])
        else
            offset += abs(mn[i]) * 1.2
            lines!(p, sol.t, offset .+ V_neuron; color=p.color[])
        end
        offset += abs(mx[i]) * 1.2
    end
    
    return p
end

@recipe(FRPlot, blox, sol) do scene
    Theme(
        color = :black,
        Axis = (
            ylabel = "Frequency (Hz)",
            xlabel = "Time (s)"
        ),
        win_size = 10, # ms
        overlap = 0,
        transient = 0
    )
end

argument_names(::Type{<: FRPlot}) = (:blox, :sol)

function Makie.plot!(p::FRPlot)
    sol = p.sol[]
    blox = p.blox[]

    ax = current_axis()
    ax.xlabel = p.Axis.xlabel[]
    ax.ylabel = p.Axis.ylabel[]

    hideydecorations!(ax)
    
    fr = firing_rate(blox, sol; win_size = p.win_size[], overlap = p.overlap[], transient = p.transient[])

    t = range(p.transient[], stop = last(sol.t), length = length(fr))
    lines!(p, t .* 1e-3, fr; color = p.color[])
    
    return p
end

function Makie.convert_arguments(::Makie.PointBased, blox::AbstractNeuronBlox, sol::AbstractSolution)
    V = voltage_timeseries(blox, sol)

    return (sol.t, V)
end

function voltage_stack(blox::Union{CompositeBlox, AbstractVector{<:AbstractBlox}}, sol::AbstractSolution; N_neurons=10, fontsize=8, color=:black)
    neurons = get_neurons(blox)
    N_ax = min(length(neurons), N_neurons)

    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Time", ylabel="Neurons")

    hideydecorations!(ax)

    stackplot!(ax, blox, sol)

    display(fig)
end

@recipe(PowerSpectrumPlot, blox, sol) do scene
    Theme(
        Axis = (
            xlabel = "Frequency (Hz)",
            ylabel = "Power Spectrum",
            xticks = [8,12,20,30, 40, 50,60,70,80,90],
            yscale = log10,
        ),
        xlims = (8, 100),
        ylims = (1e-3, 10),
        alpha_start = 8,
        beta_start = 12,
        gamma_start = 35,
        gamma_end = 100,
        alpha_label_position = (8.5, 5.0),
        beta_label_position = (22, 5.0),
        gamma_label_position = (60, 5.0),
        show_bands = true,
        sampling_rate = nothing,
        method = nothing,
        window = nothing,
        state = "V"
    )
end

argument_names(::Type{<: PowerSpectrumPlot}) = (:blox, :sol)

function Makie.plot!(p::PowerSpectrumPlot)
    sol = p.sol[]
    blox = p.blox[]

    ax = current_axis()
    xlims!(ax, p.xlims[][1], p.xlims[][2])
    ylims!(ax, p.ylims[][1], p.ylims[][2])
    ax.xlabel = p.Axis.xlabel[]
    ax.ylabel = p.Axis.ylabel[]
    ax.xticks = p.Axis.xticks[]
    ax.yscale = p.Axis.yscale[]

    if p.show_bands[]
        y1 = p.ylims[][1]
        y2 = p.ylims[][2]

        poly!(p, Point2f[(p.alpha_start[], y1), (p.alpha_start[], y2), (p.beta_start[], y2), (p.beta_start[], y1)], color = (:red,0.2), strokecolor = :black, strokewidth = 1)
        poly!(p, Point2f[(p.beta_start[], y1), (p.beta_start[], y2), (p.gamma_start[], y2), (p.gamma_start[], y1)], color = (:blue,0.2), strokecolor = :black, strokewidth = 1)
        poly!(p, Point2f[(p.gamma_start[], y1), (p.gamma_start[], y2), (p.gamma_end[], y2), (p.gamma_end[], y1)], color = (:green,0.2), strokecolor = :black, strokewidth = 1)
        
        text!(p, p.alpha_label_position[]...; text=L"\alpha", fontsize=24)
        text!(p, p.beta_label_position[]...; text=L"\beta", fontsize=24)
        text!(p, p.gamma_label_position[]...; text=L"\gamma", fontsize=24)
    end


    powspec_kwargs = (sampling_rate = p.sampling_rate[],
    method = p.method[],
    window = p.window[])

    powspec_kwargs = filter_nothing(powspec_kwargs)
    _powerspectrumplot(p, blox, sol, powspec_kwargs)

    return p
end

filter_nothing(kwargs::NamedTuple) = NamedTuple(k => v for (k, v) in pairs(kwargs) if v !== nothing)

function _powerspectrumplot(p, blox, sol::AbstractSolution, powspec_kwargs)
    powspec = powerspectrum(blox, sol, p.state[]; powspec_kwargs...)
    powerfirst = powspec.power[2]
    lines!(p, powspec.freq[2:end], powspec.power[2:end]/powerfirst)
end

function _powerspectrumplot(p, blox, sols::EnsembleSolution, powspec_kwargs)

    powspecs = powerspectrum(blox, sols, p.state[]; powspec_kwargs...)
    mean_power = mean(powspec.power[2:end] for powspec in powspecs)

    freq = powspecs[1].freq[2:end]
    std_power = std([powspec.power[2:end] for powspec in powspecs])

    y_lower = mean_power - std_power
    y_upper = mean_power + std_power
    mean_power_norm = mean_power / mean_power[1]
    y_lower = y_lower / mean_power[1]
    y_upper = y_upper / mean_power[1]

    band!(p, freq, y_lower, y_upper, color=(:purple,0.2))
    lines!(p,freq, mean_power_norm, color=:purple)
end

end