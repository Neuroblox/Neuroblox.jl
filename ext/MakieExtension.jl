module MakieExtension

isdefined(Base, :get_extension) ? using Makie : using ..Makie

using Neuroblox
using Neuroblox: CompositeBlox, meanfield_timeseries, detect_spikes, get_neurons
using SciMLBase: AbstractSolution

import Neuroblox: meanfield, meanfield!, rasterplot, rasterplot!

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

end