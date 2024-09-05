module MakieExtension

isdefined(Base, :get_extension) ? using Makie : using ..Makie

using Neuroblox
using Neuroblox: AbstractNeuronBlox, CompositeBlox
using Neuroblox: meanfield_timeseries, voltage_timeseries, detect_spikes, get_neurons
using SciMLBase: AbstractSolution

import Neuroblox: meanfield, meanfield!, rasterplot, rasterplot!, stackplot, stackplot!, voltage_stack

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

end