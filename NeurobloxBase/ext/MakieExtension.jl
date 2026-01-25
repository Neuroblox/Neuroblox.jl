module MakieExtension

isdefined(Base, :get_extension) ? using Makie : using ..Makie

using NeurobloxBase
using NeurobloxBase: AbstractBlox, AbstractNeuron, AbstractComposite
using NeurobloxBase: meanfield_timeseries, voltage_timeseries, detect_spikes, firing_rate, get_neurons, powerspectrum, strip_outer_namespace
using SciMLBase: AbstractSolution, EnsembleSolution
using LinearAlgebra: diag
using SparseArrays
using DSP
using Statistics: mean, std
using Colors: colormap

import NeurobloxBase: meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, 
                  frplot, frplot!, voltage_stack, adjacency, adjacency!
import NeurobloxBase: powerspectrumplot, powerspectrumplot!

@recipe(Adjacency, blox_or_graph) do scene
    Theme(
        colorrange = nothing,
        title = ""
    )
end

argument_names(::Type{<: Adjacency}) = (:blox_or_graph)

function Makie.plot!(p::Adjacency)
    blox_or_graph = p.blox_or_graph[]
    adj = AdjacencyMatrix(blox_or_graph)

    N = length(adj.names)
    ax_names = String.(strip_outer_namespace.(adj.names))
    ax = current_axis()
    ax.xticks = (1:N, ax_names)
    ax.yticks = (1:N, ax_names)
    ax.xticklabelrotation = pi/2
    if !isempty(p.title[])
        ax.title = p.title[]
    end 

    hidexdecorations!(ax, ticklabels = false, ticks = false)
    hideydecorations!(ax, ticklabels = false, ticks = false)

    X, Y, D = findnz(adj.matrix)
    
    colorrange = if isnothing(p.colorrange[])
        (minimum(D), maximum(D))
    else
        p.colorrange[]
    end
    cm = colormap("Grays")
    heatmap!(p, Y, X, D; colormap = cm, colorrange)

    return p
end

@recipe(MeanField, blox, sol) do scene
    Theme(
        state="V",
        xlabel = "Time (ms)",
        ylabel = "Voltage (mV)",
        title = "",
        color = :black
    )
end

argument_names(::Type{<: MeanField}) = (:blox, :sol)

function Makie.plot!(p::MeanField)
    sol = p.sol[]
    blox = p.blox[]
    state = p.state[]

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = isempty(ax.ylabel[]) ? p.ylabel[] : ax.ylabel[]
    if !isempty(p.title[])
        ax.title = p.title[]
    end 

    V = meanfield_timeseries(blox, sol, state)
    
    lines!(p, sol.t, vec(V); color=p.color[])

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
    cl = get_neuron_color(blox)
    cl isa String && (cl = [cl])

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    if !isempty(p.title[])
        ax.title = p.title[]
    end 

    spikes = detect_spikes(blox, sol; threshold=threshold)
    N_neurons = length(cl)

    for i in Base.OneTo(N_neurons)
        spike_times, _ = findnz(spikes[:, i])
        scatter!(p, sol.t[spike_times], fill(i, length(spike_times)); color=cl[i])
    end

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
    cl isa String && (cl = [cl])

    ax = current_axis()
    ax.xlabel = p.xlabel[]
    ax.ylabel = p.ylabel[]
    if !isempty(p.title[])
        ax.title = p.title[]
    end 
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
    if !isempty(p.title[])
        ax.title = p.title[]
    end 
    
    fr = firing_rate(blox, sol; win_size = p.win_size[], overlap = p.overlap[], transient = p.transient[], threshold = p.threshold[])

    t = range(p.transient[], stop = last(sol.t), length = length(fr))
    lines!(p, t .* 1e-3, fr; color = p.color[])
    
    return p
end

function Makie.convert_arguments(::Makie.PointBased, blox::AbstractNeuron, sol::AbstractSolution)
    V = voltage_timeseries(blox, sol)

    return (sol.t, V)
end

function voltage_stack(blox::Union{AbstractComposite, AbstractVector{<:AbstractBlox}}, sol::AbstractSolution; N_neurons=10, fontsize=8, color=:black)
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
        bands_generated = false,
        poly_alpha = nothing,
        poly_beta = nothing,
        poly_gamma = nothing,
        alpha_label = nothing,
        beta_label = nothing,
        gamma_label = nothing,

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
    if !isempty(p.title[])
        ax.title = p.title[]
    end 

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
    alpha_band_position = Point2f[(p.alpha_start[], y1), (p.alpha_start[], y2), (p.beta_start[], y2), (p.beta_start[], y1)]
    beta_band_position = Point2f[(p.beta_start[], y1), (p.beta_start[], y2), (p.gamma_start[], y2), (p.gamma_start[], y1)]
    gamma_band_position = Point2f[(p.gamma_start[], y1), (p.gamma_start[], y2), (p.gamma_end[], y2), (p.gamma_end[], y1)]
    labels_y = p.band_labels_vertical_position[] == "top" ? y2 - (y2-y1)*p.band_labels_vertical_offset[] : y1 + (y2-y1)*p.band_labels_vertical_offset[]

    if !p.bands_generated[]
        p.poly_alpha[] = poly!(p, alpha_band_position, color = (:red,0.2), strokecolor = :black, strokewidth = 1)
        p.poly_beta[] = poly!(p, beta_band_position, color = (:blue,0.2), strokecolor = :black, strokewidth = 1)
        p.poly_gamma[] = poly!(p, gamma_band_position, color = (:green,0.2), strokecolor = :black, strokewidth = 1)

        p.alpha_label[] = text!(p, (p.alpha_label_position[], labels_y); text=L"\alpha", fontsize=24)
        p.beta_label[] = text!(p, (p.beta_label_position[], labels_y); text=L"\beta", fontsize=24)
        p.gamma_label[] = text!(p, (p.gamma_label_position[], labels_y); text=L"\gamma", fontsize=24)
        p.bands_generated[] = true
    else
        p.poly_alpha[][1] = alpha_band_position
        p.poly_beta[][1] = beta_band_position
        p.poly_gamma[][1] = gamma_band_position
        p.alpha_label[][1] = (p.alpha_label_position[], labels_y)
        p.beta_label[][1] = (p.beta_label_position[], labels_y)
        p.gamma_label[][1] = (p.gamma_label_position[], labels_y)
    end
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
