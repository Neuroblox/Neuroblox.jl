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


import Neuroblox: meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, 
                frplot, frplot!, voltage_stack, ecbarplot, ecbarplot!, freeenergy, freeenergy!,
                adjacency, adjacency!
import Neuroblox: powerspectrumplot, powerspectrumplot!

@recipe(Adjacency, blox_or_graph) do scene
    Theme(
        colormap = :vanimo
    )
end

argument_names(::Type{<: Adjacency}) = (:blox_or_graph)

function Makie.plot!(p::Adjacency)
    blox_or_graph = p.blox_or_graph[]
    adj = AdjacencyMatrix(blox_or_graph)

    N = length(adj.names)

    ax = current_axis()
    ax.xticks = (Base.OneTo(N), String.(adj.names))
    ax.yticks = (Base.OneTo(N), String.(adj.names))
    ax.xticklabelrotation = pi/2

    hidexdecorations!(ax, ticklabels = false, ticks = false)
    hideydecorations!(ax, ticklabels = false, ticks = false)

    X, Y, D = findnz(adj.matrix)

    heatmap!(p, X, Y, D; colormap = p.colormap[], colorrange = (minimum(D), maximum(D)))

    return p
end

@recipe(FreeEnergy, spDCMresults) do scene
    Theme(
        xlabel = "Iterations",
        ylabel = "Free Energy",
        title = ""
    )
end

argument_names(::Type{<: FreeEnergy}) = (:spDCMresults)

function Makie.plot!(p::FreeEnergy)
    F = p.spDCMresults[].F
    deleteat!(F, 1)   # remove the first value since that's always -Inf
    
    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]

    lines!(p, 1:length(F), F)
    scatter!(p, 1:length(F), F)
    return p
end

@recipe(ECBarPlot, spDCMresults, spDCMsetup, groundtruth) do scene
    Theme(
        xlabel = "Parameter Name",
        ylabel = "Effective Connectivity",
        title = ""
    )
end

argument_names(::Type{<: ECBarPlot}) = (:spDCMresults, :spDCMsetup, :groundtruth)

function Makie.plot!(p::ECBarPlot)
    nr = p.spDCMsetup[].systemnums[1]  # number of regions
    diagidx = 1:(nr+1):nr^2
    modelparam = p.spDCMsetup[].modelparam
    idx = collect(1:nr^2)
    deleteat!(idx, diagidx)
    xlabels = string.(collect(keys(modelparam))[idx])
    ax = current_axis()
    ax.xticks = (1:(nr^2-nr), xlabels)
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]

    gt = copy(vec(p.groundtruth[]))   # get ground truth values
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

@recipe(MeanField, blox, sol) do scene
    Theme(
        xlabel = "Time (ms)",
        ylabel = "Voltage (mV)",
        title = ""
    )
end

argument_names(::Type{<: MeanField}) = (:blox, :sol)

function Makie.plot!(p::MeanField)
    sol = p.sol[]
    blox = p.blox[]

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]

    V = meanfield_timeseries(blox, sol)
    
    lines!(p, sol.t, vec(V))

    return p
end

@recipe(RasterPlot, blox, sol) do scene
    Theme(
        color = :black,
        threshold = nothing,
        xlabel = "Time (ms)",
        ylabel = "Neurons",
        title = ""
    )
end

argument_names(::Type{<: RasterPlot}) = (:blox, :sol)

function Makie.plot!(p::RasterPlot)
    sol = p.sol[]
    t = sol.t
    blox = p.blox[]
    threshold = p.threshold[]

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]

    spikes = detect_spikes(blox, sol; threshold=threshold)
    spike_times, neuron_indices = findnz(spikes)
    scatter!(p, sol.t[spike_times], neuron_indices; color=p.color[])

    return p
end

@recipe(StackPlot, blox, sol) do scene
    Theme(
        dynamic_gap = false,        
        xlabel = "Time (ms)",
        ylabel = "Neurons",
        title = ""
        )
end

argument_names(::Type{<: StackPlot}) = (:blox, :sol)

function Makie.plot!(p::StackPlot)
    sol = p.sol[]
    blox = p.blox[]
    
    cl = get_neuron_color(blox)

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]
    hideydecorations!(ax; label = false)

    V = voltage_timeseries(blox, sol)
    
    V = V .- mean(V; dims = 1)

    mx = maximum(V; dims = 1)
    mn = minimum(V; dims = 1)


    
    if p.dynamic_gap[]
        offset = 0.0
        for (i, V_neuron) in enumerate(eachcol(V))
            if i == 1
                lines!(p, sol.t, V_neuron, color=cl[i])
            else
                offset += abs(mn[i]) * 1.2
                lines!(p, sol.t, offset .+ V_neuron,color=cl[i])
            end
            offset += abs(mx[i]) * 1.2
        end
    else
        offset = maximum(mx .- mn) * 1.2
        for (i, V_neuron) in enumerate(eachcol(V))
            lines!(p, sol.t, (i - 1) * offset .+ V_neuron, color=cl[i])
        end
    end
    
    return p
end

@recipe(FRPlot, blox, sol) do scene
    Theme(
        color = :black,
        xlabel = "Time (s)",
        ylabel = "Frequency (Hz)",
        title = "",
        win_size = 10, # ms
        overlap = 0,
        transient = 0,
        threshold = nothing
    )
end

argument_names(::Type{<: FRPlot}) = (:blox, :sol)

function Makie.plot!(p::FRPlot)
    sol = p.sol[]
    blox = p.blox[]

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.title = p.title[]

    hideydecorations!(ax)
    
    fr = firing_rate(blox, sol; win_size = p.win_size[], overlap = p.overlap[], transient = p.transient[], threshold = p.threshold[])

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
    ax = Axis(fig[1,1], xlabel="Time (ms)", ylabel="Neurons")

    hideydecorations!(ax)

    stackplot!(ax, blox, sol)

    display(fig)
end

@recipe(PowerSpectrumPlot, blox, sol) do scene
    Theme(
        xlabel = "Frequency (Hz)",
        ylabel = "Power Spectrum (dB)",
        xticks = [8,12,20,30, 40, 50,60,70,80,90],
        yscale = identity,
        title = "",
        xlims = (8, 100),
        ylims = nothing,
        alpha_start = 8,
        beta_start = 12,
        gamma_start = 35,
        gamma_end = 100,

        alpha_label_position = 8.5,
        beta_label_position = 22,
        gamma_label_position = 60,
        band_labels_vertical_position = "top",
        band_labels_vertical_offset = 0.065,

        show_bands = true,
        sampling_rate = nothing,
        method = nothing,
        window = nothing,
        state = "V"
    )
end

argument_names(::Type{<: PowerSpectrumPlot}) = (:blox, :sol)

function Makie.plot!(p::PowerSpectrumPlot)
    set_powerplot_axis(p)

    sol = p.sol[]
    blox = p.blox[]

    powspec_kwargs = (sampling_rate = p.sampling_rate[],
                        method = p.method[],
                        window = p.window[])

    powspec_kwargs = filter_nothing(powspec_kwargs)
    _powerspectrumplot(p, blox, sol, powspec_kwargs)

    return p
end

function filter_data_within_limits(x, y, xlims)
    mask = (x .>= xlims[1]) .& (x .<= xlims[2])
    return x[mask], y[mask]
end

filter_nothing(kwargs::NamedTuple) = NamedTuple(k => v for (k, v) in pairs(kwargs) if v !== nothing)

const PreComputedPowerSpectrums = PowerSpectrumPlot{<:Tuple{<:Vector{<:DSP.Periodograms.Periodogram}}}

function Makie.plot!(p::PreComputedPowerSpectrums)
    spectra = p[1][]
    
    # plot only within x-limits so that Makie computes the right limits
    xlims = p.xlims[]
    in_range = findall(xlims[1]-1 .<= spectra[1].freq .<= xlims[2]+1)

    for powspec in spectra
        power_db = 10 * log10.(powspec.power[in_range]) # convert to dB scale
        power_db .-= power_db[1]
        lines!(p, powspec.freq[in_range], power_db)
    end

    set_powerplot_axis(p)
    return p
end

const PreComputedPowerSpectrum = PowerSpectrumPlot{<:Tuple{<:DSP.Periodograms.Periodogram}}

function Makie.plot!(p::PreComputedPowerSpectrum)
    spectra = p[1][]
    # plot only within x-limits so that Makie computes the right limits
    xlims = p.xlims[]
    in_range = findall(xlims[1]-1 .<= spectra.freq .<= xlims[2]+1)

    power_db = 10 * log10.(spectra.power[in_range]) # convert to dB scale
    power_db .-= power_db[1]
    lines!(p, spectra.freq[in_range], power_db)
    set_powerplot_axis(p)
    return p
end

function set_powerplot_axis(p)

    ax = current_axis()
    xlims!(ax, p.xlims[][1], p.xlims[][2])
    !isnothing(p.ylims[]) && ylims!(ax, p.ylims[][1], p.ylims[][2])

    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    ax.xticks = p.xticks[]
    ax.yscale = p.yscale[]
    ax.title = p.title[]

    if p.show_bands[]
        # if the y-limits are not provided by the user, Makie computes them but only after the plot has been displayed
        if isnothing(p.ylims[])
            on(ax.finallimits) do lims
                y1, y2 = minimum(lims)[2], maximum(lims)[2]
                show_bands!(p, y1, y2)
            end
        else
            y1, y2 = p.ylims[][1], p.ylims[][2]
            show_bands!(p, y1, y2)
        end
    end
end

function show_bands!(p, y1, y2)
    poly!(p, Point2f[(p.alpha_start[], y1), (p.alpha_start[], y2), (p.beta_start[], y2), (p.beta_start[], y1)], color = (:red,0.2), strokecolor = :black, strokewidth = 1)
    poly!(p, Point2f[(p.beta_start[], y1), (p.beta_start[], y2), (p.gamma_start[], y2), (p.gamma_start[], y1)], color = (:blue,0.2), strokecolor = :black, strokewidth = 1)
    poly!(p, Point2f[(p.gamma_start[], y1), (p.gamma_start[], y2), (p.gamma_end[], y2), (p.gamma_end[], y1)], color = (:green,0.2), strokecolor = :black, strokewidth = 1)

    labels_y = p.band_labels_vertical_position[] == "top" ? y2 - (y2-y1)*p.band_labels_vertical_offset[] : y1 + (y2-y1)*p.band_labels_vertical_offset[]
    text!(p, (p.alpha_label_position[], labels_y); text=L"\alpha", fontsize=24)
    text!(p, (p.beta_label_position[], labels_y); text=L"\beta", fontsize=24)
    text!(p, (p.gamma_label_position[], labels_y); text=L"\gamma", fontsize=24)
end

function _powerspectrumplot(p, blox, sol::AbstractSolution, powspec_kwargs)
    powspec = powerspectrum(blox, sol, p.state[]; powspec_kwargs...)
    xlims = p.xlims[]
    in_range = findall(xlims[1]-1 .<= powspec.freq .<= xlims[2]+1)

    power_db = 10 * log10.(powspec.power[in_range]) # convert to dB scale
    power_db .-= power_db[1]
    lines!(p, powspec.freq[in_range], power_db)
end

function _powerspectrumplot(p, blox, sols::EnsembleSolution, powspec_kwargs)

    powspecs = powerspectrum(blox, sols, p.state[]; powspec_kwargs...)

    xlims = p.xlims[]
    in_range = findall(xlims[1]-1 .<= powspecs[1].freq .<= xlims[2]+1)

    mean_power = mean(powspec.power[in_range] for powspec in powspecs)
    std_power = std([powspec.power[in_range] for powspec in powspecs])

    power_db = 10 * log10.(mean_power)
    power_db .-= power_db[1]
    ribbon_dB = 10*log10.(1 .+ std_power ./ mean_power)
    y_lower = power_db - ribbon_dB
    y_upper = power_db + ribbon_dB
    freq = powspecs[1].freq[in_range]

    band!(p, freq, y_lower, y_upper, color=(:purple,0.2))
    lines!(p,freq, power_db, color=:purple)
end

end