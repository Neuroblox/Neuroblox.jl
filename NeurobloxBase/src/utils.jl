macro named(ex)
    ex_out = @match ex begin
        :($name = $f($(Expr(:parameters, kwargs...)), $(args...))) => :($name = $f($(args...); name=$(QuoteNode(name)), $(kwargs...) ))
        :($name = $f($(args...))) => :($name = $f($(args...); name=$(QuoteNode(name))))
        ex => error("Malformed expression passed to @named, must be of the form `name = f(args...; kwargs...)`, got\n $ex")
    end
    esc(ex_out)
end

"""
    get_exci_neurons(n::Union{GraphSystem, AbstractBlox})

Returns all excitatory neurons within a blox or graph.
- If `n <: Union{GraphSystem, AbstractComposite}`, returns all excitatory neurons within the graph or composite blox.
- If `n::AbstractExciNeuron`, returns a single-element vector `[n]`.
"""
get_exci_neurons(n::AbstractExciNeuron) = [n]
get_exci_neurons(n) = AbstractExciNeuron[]

function get_exci_neurons(g::GraphSystem)
    mapreduce(x -> get_exci_neurons(x), vcat, nodes(g.flat_graph))
end

function get_exci_neurons(b::AbstractComposite)
    mapreduce(x -> get_exci_neurons(x), vcat, nodes(get_flat_graph(b)))
end

"""
    get_inh_neurons(n::Union{GraphSystem, AbstractBlox})

Returns all inhibitory neurons within a blox or graph.
- If `n <: Union{GraphSystem, AbstractComposite}`, returns all excitatory neurons within the graph or composite blox.
- If `n::AbstractExciNeuron`, returns a single-element vector `[n]`.
"""
get_inh_neurons(n::AbstractInhNeuron) = [n]
get_inh_neurons(n) = AbstractInhNeuron[]

function get_inh_neurons(g::GraphSystem)
    mapreduce(x -> get_inh_neurons(x), vcat, nodes(g.flat_graph))
end

function get_inh_neurons(b::AbstractComposite)
    mapreduce(x -> get_inh_neurons(x), vcat, nodes(get_flat_graph(b)))
end

get_neurons(n::AbstractNeuron) = [n]
get_neurons(n) = AbstractNeuron[]

function get_neurons(b::AbstractComposite)
    mapreduce(x -> get_neurons(x), vcat, nodes(get_flat_graph(b)))
end

function get_neurons(vn::AbstractVector{<:AbstractBlox})
    mapreduce(x -> get_neurons(x), vcat, vn)
end

get_parts(blox::AbstractComposite) = collect(nodes(get_flat_graph(blox)))
get_parts(blox::Union{AbstractBlox, AbstractObserver}) = blox

get_components(blox::AbstractComposite) = mapreduce(get_components, vcat, get_parts(blox))
get_components(blox::Vector{<:AbstractBlox}) = mapreduce(get_components, vcat, blox)
get_components(blox) = [blox]

get_dynamics_components(blox::AbstractDiscrete) = []
get_dynamics_components(blox::Union{AbstractNeuralMass, AbstractNeuron, AbstractReceptor}) = [blox]
get_dynamics_components(blox::AbstractComposite) = mapreduce(get_dynamics_components, vcat, get_parts(blox))
get_dynamics_components(blox::Vector{<:AbstractBlox}) = mapreduce(get_dynamics_components, vcat, blox)

get_neuron_color(n::AbstractExciNeuron) = "blue"
get_neuron_color(n::AbstractInhNeuron) = "red"
get_neuron_color(n::AbstractNeuron) = "black"

get_neuron_color(n::Union{AbstractComposite, Vector{<:AbstractBlox}}) = map(get_neuron_color, get_neurons(n))

function get_discrete_parts(b::AbstractComposite)
    mapreduce(x -> get_discrete_parts(x), vcat, nodes(get_flat_graph(b)))
end

function get_gap(kwargs, name_blox1, name_blox2)
    get(kwargs, :gap, false)    
end

function get_gap_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :gap_weight) do
        error("Gap junction weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

"""
    nameof(blox)

Returns the name of `blox` without the namespace prefix. 

See also [`namespaced_nameof`](@ref) and [`full_namespaced_nameof`](@ref). 
"""
nameof(blox::Union{AbstractObserver, AbstractBlox, AbstractComposite, AbstractActionSelection}) = getfield(blox, :name)
namespaceof(blox) = blox.namespace

"""
    namespaced_nameof(blox)

Returns the name of the blox, prefaced by its entire inner namespace (all levels of the namespace EXCEPT for 
the highest level). Namespaces represent the parent composite blox or graph that `blox` belongs to.
E.g. the name `cort1.wta2.neuron4` comes from a neuron `neuron4` which belongs to `wta2` (possibly a `WinnerTakeAll` blox) which in turn is part of `cort1` (possibly a `Cortical` blox).

See also [`nameof`](@ref) and [`full_namespaced_nameof`](@ref).
"""
namespaced_nameof(blox) = namespaced_name(inner_namespaceof(blox), nameof(blox))

"""
    full_namespaced_nameof(blox)

Return the name of the blox, prefaced by its entire inner namespace (all levels of the namespace INCLUDING for 
the highest level). Namespaces represent the parent composite blox or graph that `blox` belongs to.
E.g. the name `mdl.cort1.wta2.neuron4` comes from a neuron `neuron4` which belongs to `wta2` (possibly a `WinnerTakeAll` blox), which in turn is part of `cort1` (possibly a `Cortical` blox), which finally is part of a circuit `mdl`.

See also [`nameof`](@ref) and [`namespaced_nameof`](@ref).
"""
full_namespaced_nameof(blox) = namespaced_name(namespaceof(blox), nameof(blox))

"""
    inner_namespaceof(blox)

Returns the complete namespace EXCLUDING the outermost (highest) level.
This is useful for manually preparing equations (e.g. connections, see Connector),
that will later be composed and will automatically get the outermost namespace.
""" 
function inner_namespaceof(blox)
    # DO we still need this?
    parts = split((string ∘ namespaceof)(blox), '₊')
    if length(parts) == 1
        return nothing
    else
        return join(parts[2:end], '₊')
    end
end

"""
    strip_outer_namespace(s::Symbol)

Strip the outermost namespace from a namespaced name, if there is a namespace. This should have the property that

```julia
strip_outer_namespace(full_namespaced_nameof(blox)) == namespaced_nameof(blox)
```
"""
function strip_outer_namespace(s::Symbol)
    parts = split(String(s), '₊')
    if length(parts) == 1
        return s
    else
        return Symbol(join(parts[2:end], '₊'))
    end
end

"""
    namespaced_name(parent_name, name)

Return a symbol consisting of `parent_name` joined to `name` via ₊.
"""
namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(::Nothing, name) = Symbol(name)

function outputs end
outputs(::Subsystem) = NamedTuple()

output(x) = only(outputs(x))

function inputs end

get_graph(blox::AbstractComposite) = blox.graph
get_graph(blox) = let g = GraphSystem()
    add_node!(g, blox)
    g
end

get_flat_graph(x) = get_graph(x).flat_graph

"""
    get_weight(kwargs, name_blox1, name_blox2)

Obtain the value of the connection weight between two Blox from a list of kwargs. Error if not found.
"""
function get_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :weight) do
        @info "Connection weight from $name_blox1 to $name_blox2 is not specified. Assuming weight=1" 
        return 1.0
    end
end

"""
    get_event_time(kwargs, name_blox1, name_blox2)

Obtain the value of the time of an event between two blox from a list of kwargs. Error if not found.
"""
function get_event_time(kwargs, name_blox1, name_blox2)
    t_event = get(kwargs, :t_event) do
        error("Time for the event that affects the connection from $name_blox1 to $name_blox2 is not specified.")
    end
    float(t_event)
end

"""
    get_weightmatrix(kwargs, name_blox1, name_blox2)

Obtain the value of weight matrix for connections between two neural populations from a list of kwargs. 
Error if not found.
"""
function get_weightmatrix(kwargs, name_blox1, name_blox2)
    get(kwargs, :weightmatrix) do
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_delay(kwargs, name_blox1, name_blox2)
    get(kwargs, :delay) do 
#        @debug "Delay constant from $name_blox1 to $name_blox2 is not specified. It is assumed that there is no delay."
        return 0
    end
end

"""
    get_density(kwargs, name_blox1, name_blox2)

Obtain the value of connection density between two neural populations from a list of kwargs. Error if not found.
"""
function get_density(kwargs, name_blox1, name_blox2)
    get(kwargs, :density) do 
        error("Connection density from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_connection_matrix(kwargs, name_out, name_in, N_out, N_in)
    sz = (N_out, N_in)
    connection_matrix = get(kwargs, :connection_matrix) do
        density = get_density(kwargs, name_out, name_in)
        dist = Bernoulli(density)
        rng = get(kwargs, :rng, default_rng())
        rand(rng, dist, sz...)
    end
    if size(connection_matrix) != sz
        error(ArgumentError("The supplied connection matrix between $(name_out) and $(name_in) is an "
                            * "incorrect size. Got $(size(connection_matrix)), whereas $(name_out) has "
                            * "$N_out excitatory neurons, and $name_in has $N_in excitatory neurons."))
    end
    if eltype(connection_matrix) != Bool
        error(ArgumentError("The supplied connection matrix between $(name_out) and $(name_in) must "
                            * "be an array of Bool, got $(eltype(connection_matrix)) instead."))
    end
    connection_matrix
end

function get_learning_rule(kwargs, name_src, name_dest)
    if haskey(kwargs, :learning_rule)
        return deepcopy(kwargs[:learning_rule])
    else
        return NoLearningRule()
    end
end

function get_connection_rule(kwargs, blox_src::AbstractComposite, blox_dst::AbstractComposite)
    wm = get(kwargs, :weightmatrix, nothing)
    if !isnothing(wm)
        @info "Weight matrix passed in for connection between $(namespaced_nameof(blox_src)) and $(namespaced_nameof(blox_dst)). Defaulting to a weight matrix connection."
        return :weightmatrix
    end

    cr = get(kwargs, :connection_rule) do
        @info "Connection rule from $(namespaced_nameof(blox_src)) to $(namespaced_nameof(blox_dst)) not specified. Defaulting to a hypergeometric connection."
        :hypergeometric
    end
    cr = Symbol(cr) # in case of String kwarg
    if (cr == :hypergeometric) || (cr == :density) || (cr == :weightmatrix)
        return cr
    else
        @error "Connection rule not recognized. Available options are $("hypergeometric"), $("weightmatrix"), and $("density")."
    end
end

function get_connection_rule(kwargs, bloxout::Union{AbstractNeuron, AbstractNeuralMass}, bloxin::Union{AbstractNeuron, AbstractNeuralMass}, w)
    name_blox1 = nameof(bloxout)
    name_blox2 = nameof(bloxin)
    cr = get(kwargs, :connection_rule) do
        @info "Neuron connection rule from $name_blox1 to $name_blox2 is not specified. It is assumed that there is a basic weighted connection."
        cr = "basic"
    end
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

     # Logic based on connection rule type
     if isequal(cr, "basic")
        outs = outputs(bloxout; namespaced=true)
        if !isempty(outs)
            x = first(outs)
            rhs = x*w
            length(outs) > 1 && @info "Blox $name_blox1 has more than one outputs. Defaulting to output=$x"
        else
            error("Blox $name_blox1 has no outputs. Please assign [output=true] to the variables you want to use as outputs or write a dispatch for connection_equations.")
        end
    elseif isequal(cr, "psp")
        rhs = w*sys_out.G*(sys_out.E_syn - sys_in.V)
    else
        error("Connection rule not recognized")
    end

    return rhs
end

to_vector(v::AbstractVector) = v
to_vector(v) = [v]

nanmean(x) = mean(filter(!isnan,x))

function replace_refractory!(V, blox::AbstractComposite, sol::SciMLBase.AbstractSolution)
    neurons = get_neurons(blox)
    for (i, n) in enumerate(neurons)
        V[:, i] = replace_refractory!(V[:,i], n, sol)
    end
end

replace_refractory!(V, blox, sol::SciMLBase.AbstractSolution) = V

function find_spikes(x::AbstractVector{T}; threshold=zero(T)) where {T}    
    spike_idxs = argmaxima(x)
    peakheights!(spike_idxs, x[spike_idxs]; min = threshold)
    
    spikes = sparsevec(spike_idxs, ones(length(spike_idxs)), length(x))

    return spikes
end


function count_spikes(x::AbstractVector{T}; threshold=zero(T)) where {T}
    spikes = find_spikes(x; threshold)
    
    return nnz(spikes)
end

"""
    detect_spikes(blox::AbstractNeuron, sol::AbstractSolution;
                  threshold=nothing,
                  tolerance=1e-3,
                  ts=nothing,
                  scheduler=:serial)

Find the spike timepoints of `blox` according to simulation solution `sol`. 
Return a SparseVector that is equal to 1 at the timepoints of each spike.

Keyword arguments:
    - threshold : [mV] Spiking threshold. 
                Note that neurons like [`NeurobloxPharma.HHNeuronExci`](@ref) and [`NeurobloxPharma.HHNeuronInhib`](@ref) do not inherently contain a threshold and so require this `threshold` argument to be passed in order to determine spiking events. 
    - tolerance : [mV] The tolerance around the threshold value in which a maximum counts as a spike.
    - ts : [ms] Timepoints in simulation solution `sol` where spikes are detected. It can be a range, a vector or an individual timepoint. 
            If no values are passed (and `ts=nothing`) then the entire range of the solution `sol` is used. 
"""
function detect_spikes(
    blox::AbstractNeuron, sol::SciMLBase.AbstractSolution; 
    threshold = nothing, tolerance = 1e-3, ts = nothing, scheduler=:serial
)
    namespaced_name = full_namespaced_nameof(blox)

    thrs_value = if isnothing(threshold)
        threshold_param_name = Symbol(namespaced_name, "₊θ")

        get_thrs = getp(sol, threshold_param_name)
        get_thrs(sol)
    else
        threshold
    end
    
    V = voltage_timeseries(blox, sol; ts)
    spikes = find_spikes(V; threshold = thrs_value - tolerance)

    return spikes
end

function detect_spikes(
    blox::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}}, sol::SciMLBase.AbstractSolution;
    threshold = nothing, ts=nothing, scheduler=:serial, kwargs...
)

    neurons = get_neurons(blox)

    S = tmapreduce(sparse_hcat, neurons; scheduler, kwargs...) do neuron
        detect_spikes(neuron, sol; threshold, ts)
    end

    return S
end

"""
    firing_rate(blox::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution;
                transient=0, win_size=last(sol.t) - transient,
                overlap=0, threshold=nothing,
                scheduler=:serial, kwargs...)

Calculates the firing rate of `blox`. If `blox` is a composite blox or a vector containing multiple elements, then the average firing rate across all neurons in `blox` is returned.

Keyword arguments:
- win_size : [ms] Sliding window size.
- overlap : in range [0,1]. Overlap between two consecutive sliding windows.
- transient : [ms] Transient period in the beginning of the timeseries that is ignored during firing rate calculation.
- threshold : [mV] Spiking threshold.

See also [`frplot`](@ref).
"""
function firing_rate(
    blox, sol::SciMLBase.AbstractSolution; 
    transient = 0, win_size = last(sol.t) - transient, overlap = 0, 
    threshold = nothing, scheduler=:serial, kwargs...)

    spikes = detect_spikes(blox, sol; threshold, scheduler, kwargs...)
    N_neurons = size(spikes, 2)
    
    ts = sol.t
    t_win_start = transient:(win_size - win_size*overlap):(last(ts) - win_size)

    fr = map(t_win_start) do tws
        idx_start = findfirst(x -> x >= tws, ts)
        idx_end = findfirst(x -> x >= tws + win_size, ts)
        
        1000.0 * (nnz(spikes[idx_start:idx_end, :]) / N_neurons) / win_size
    end

    return fr
end

function firing_rate(
    blox, sols::SciMLBase.EnsembleSolution; 
    transient = 0, win_size = last(sols.u[1].t) - transient, overlap = 0,
    threshold = nothing, scheduler=:serial, kwargs...)

    firing_rates = map(sols) do sol
        firing_rate(blox, sol; transient, win_size, overlap, threshold, scheduler, kwargs...)
    end

    mean_fr = mean(firing_rates)
    std_fr = std(firing_rates)

    return mean_fr, std_fr
end

"""
    inter_spike_intervals(blox::AbstractNeuron, sol; threshold, ts)

Return the time intervals between subsequent spikes in the solution of a single neuron.
"""
function inter_spike_intervals(
    blox::AbstractNeuron, sol::SciMLBase.AbstractSolution; 
    threshold = nothing, ts=nothing
)
    spikes = detect_spikes(blox, sol; threshold, ts)
    ISI = diff(sol.t[spikes.nzind])

    return ISI
end
"""
    inter_spike_intervals(blox::AbstractNeuron, sol; threshold, ts)

Return the time intervals between subsequent spikes in the solution of a Blox.
Outputs a matrix whose rows are the interspike intervals for a single neuron.
"""
function inter_spike_intervals(
    blox::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}}, sol::SciMLBase.AbstractSolution;
    threshold = nothing, ts=nothing, scheduler=:serial, kwargs...
)

    neurons = get_neurons(blox)

    ISIs = tmapreduce(sparse_hcat, neurons; scheduler, kwargs...) do neuron
        inter_spike_intervals(neuron, sol; threshold, ts)
    end

    return ISIs
end

"""
    flat_inter_spike_intervals(blox::AbstractNeuron, sol; threshold, ts)

Return the time intervals between subsequent spikes in the solution of a Blox.
Concatenates the lists of interspike intervals of all vectors into a single vector.
"""
function flat_inter_spike_intervals(
    blox::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}}, sol::SciMLBase.AbstractSolution;
    threshold = nothing, ts=nothing, scheduler=:serial, kwargs...
)

    neurons = get_neurons(blox)

    ISIs = tmapreduce(sparse_vcat, neurons; scheduler, kwargs...) do neuron
        inter_spike_intervals(neuron, sol; threshold, ts)
    end

    return ISIs
end

"""
    state_timeseries(blox::AbstractBlox, sol::SciMLBase.AbstractSolution, state::String; ts = nothing)

Return the timeseries of the state variable `state` from `blox`. 

Keyword arguments:
- ts : [ms] Timepoints at which the timeseries is evaluated in simulation solution `sol`. It can be a range, a vector or an individual timepoint.
        If no values are passed (and `ts=nothing`) then the entire range of the solution `sol` is used.
"""
function state_timeseries(blox, sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
    namespaced_name = full_namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")

    if isnothing(ts)
        return sol[state_name]
    else
        return Array(sol(ts, idxs = state_name))
    end
end

"""
    state_timeseries(cb::Union{AbstractComposite, Vector{<:AbstractBlox}}, 
                     sol::SciMLBase.AbstractSolution, state::String; ts = nothing)

Return the timeseries of the state variable `state` for each blox contained in `cb`. 
The resulting collection of timeseries are stacked in a Matrix where each row is a separate blox in `cb`.

Keyword arguments:
- ts : [ms] Timepoints at which the timeseries is evaluated in simulation solution `sol`. It can be a range, a vector or an individual timepoint.
        If no values are passed (and `ts=nothing`) then the entire range of the solution `sol` is used.
"""
function state_timeseries(cb::Union{AbstractComposite, AbstractVector{<:AbstractBlox}},
                          sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
    neurons = [blox for blox in get_dynamics_components(cb) if Symbol(state) ∈ state_symbols(typeof(blox))]
    state_names = map(neuron -> Symbol(full_namespaced_nameof(neuron), "₊", state), neurons)

    if isnothing(ts)
        s = stack(sol[state_names], dims=1)
    else
        s = transpose(Array(sol(ts; idxs=state_names)))
    end
    return s
end

"""
    meanfield_timeseries(cb::Union{AbstractComposite, Vector{AbstractNeuron}, sol, state; ts)

Return the average timeseries of state variable `state` over all neurons contained in `cb`. 

Keyword arguments:
- ts : [ms] Timepoints at which the timeseries is evaluated in simulation solution `sol`. It can be a range, a vector or an individual timepoint.
        If no values are passed (and `ts=nothing`) then the entire range of the solution `sol` is used.
"""
function meanfield_timeseries(cb::Union{AbstractComposite, AbstractVector{<:Union{AbstractNeuron, AbstractReceptor}}},
                              sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
                              
    s = state_timeseries(cb, sol, state; ts)

    return vec(mapslices(nanmean, s; dims = 2))
end

voltage_timeseries(blox, sol::SciMLBase.AbstractSolution; ts=nothing) = 
    state_timeseries(blox, sol, "V"; ts)

"""
    voltage_timeseries(cb::Union{AbstractComposite, Vector{Union{AbstractNeuron, AbstractNeuralMass}}, sol::AbstractSolution; ts)

Return the voltage timeseries of all neurons and/or neural masses contained in `blox`. 
The output is a Matrix where each row is a separate neuron or neural mass and each column is a timepoint.

Keyword arguments:
- ts : [ms] Timepoints at which the voltage values are evaluated in simulation solution `sol`. It can be a range, a vector or an individual timepoint.
        If no values are passed (and `ts=nothing`) then the entire range of the solution `sol` is used.
"""
function voltage_timeseries(cb::Union{AbstractComposite, AbstractVector{<:AbstractBlox}},
                            sol::SciMLBase.AbstractSolution; ts=nothing)
    return state_timeseries(cb, sol, "V"; ts)
end

function meanfield_timeseries(cb::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}},
                              sol::SciMLBase.AbstractSolution; ts=nothing)
    V = voltage_timeseries(cb, sol; ts)
    replace_refractory!(V, cb, sol)

    return vec(mapslices(nanmean, V; dims = 2))
end

function powerspectrum(cb::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}},
                       sol::SciMLBase.AbstractSolution, state::String; sampling_rate=nothing,
                       method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    s = meanfield_timeseries(cb, sol, state; ts = t_sampled)

    return method(s, fs=sampling_freq, window=window)
end

"""
    powerspectrum(cb::Union{AbstractComposite, Vector{AbstractNeuron}}, sol::AbstractSolution; sampling_rate=nothing, method=periodogram, window=nothing)

Calculate the power spectrum of the voltage timeseries for all neurons and/or neural masses contained in `blox`.

See [`powerspectrumplot`](@ref) for more information on the keyword arguments.
"""
function powerspectrum(cb::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}},
                       sol::SciMLBase.AbstractSolution; sampling_rate=nothing,
                       method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    V = voltage_timeseries(cb, sol; ts = t_sampled)
    replace_refractory!(V, cb, sol)

    return method(vec(mapslices(nanmean, V; dims = 2)), fs = sampling_freq, window=window)
end

function powerspectrum(blox::AbstractNeuron, sol::SciMLBase.AbstractSolution, state::String;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    namespaced_name = namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    data = isnothing(t_sampled) ? sol[state_name] : Array(sol(t_sampled, idxs = state_name))

    return method(data, fs = sampling_freq, window=window)
end

function powerspectrum(cb::Union{AbstractComposite, AbstractVector{<:AbstractNeuron}},
                       sols::SciMLBase.EnsembleSolution, state::String; sampling_rate=nothing,
                       method=periodogram, window=nothing)::Vector{DSP.Periodograms.Periodogram}

    t_sampled, sampling_freq = get_sampling_info(sols[1]; sampling_rate=sampling_rate)
    powspecs = DSP.Periodograms.Periodogram[]
    
    for sol in sols
        s = meanfield_timeseries(cb, sol, state; ts = t_sampled)
        powspec = method(s, fs=sampling_freq, window=window)
        push!(powspecs, powspec)
    end

    return powspecs
end

function powerspectrum(cb::AbstractNeuralMass, sol::SciMLBase.AbstractSolution, state::String;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    namespaced_name = namespaced_nameof(cb)
    state_name = Symbol(namespaced_name, "₊$(state)")

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    data = isnothing(t_sampled) ? sol[state_name] : Array(sol(t_sampled, idxs = state_name))

    return method(data, fs = sampling_freq, window=window)
end

function get_sampling_info(sol::SciMLBase.AbstractSolution; sampling_rate=nothing)
    t_raw = unique(sol.t)
    dt = diff(t_raw)
    dt_std = std(dt)
    first_diff = dt[1]

    # check if the solution was saved at regular time steps
    if !isapprox(dt_std, 0, atol=1e-2 * first_diff)
        if isnothing(sampling_rate)
            @warn("Solution not saved at fixed time steps. Provide 'sampling_rate' in milliseconds.")
            sampling_rate = first_diff
        end
        t_sampled = t_raw[1]:sampling_rate:t_raw[end]
        return t_sampled, 1000 / sampling_rate
    else
        sampling_rate = first_diff
        return nothing, 1000 / sampling_rate
    end
end


"""
    params_with_substrings(sys, substrs::Union{Symbol, String, Vector{Symbol}, Vector{String}}; intersection = true)

Return the set of parameter names and parameter indices for parameters whose names include the substrings in `substrs`. If `intersection` is set, then return only parameters whose names contain ALL of the substrings; otherwise return parameters whose names include ANY of the substrings.
"""
function params_with_substrings(obj, substrs::Union{String, Symbol, Vector{String}, Vector{Symbol}, Tuple}; intersection = true)
    p_dict = if obj isa GraphSystemParameters
        obj.symbolic_indexing_namemap.param_namemap
    elseif obj isa ODEProblem
        obj.p.symbolic_indexing_namemap.param_namemap
    else
        error("Parameter search only works with systems with symbolic parameters.")
    end
    idxs_with_substrings(p_dict, substrs; intersection)
end

"""
    states_with_substrings(sys, substrs::Union{Symbol, String, Vector{Symbol}, Vector{String}}; intersection = true)

Return the set of state names and state indices for states whose names include the substrings in `substrs`. If `intersection` is set, then return only states whose names contain ALL of the substrings; otherwise return states whose names include ANY of the substrings.
"""
function states_with_substrings(obj, substrs::Union{String, Symbol, Vector{Symbol}, Vector{String}, Tuple}; intersection = true)
    u_dict = if obj isa GraphSystemParameters
         obj.symbolic_indexing_namemap.state_namemap
    elseif obj isa ODEProblem
         obj.symbolic_indexing_namemap.state_namemap
    else
        error("State search by substring only works with systems with symbolic states.")
    end
    idxs_with_substrings(u_dict, substrs; intersection)
end

function idxs_with_substrings(dict::OrderedDict, substrs::Union{String, Symbol, Vector{String}, Vector{Symbol}, Tuple}; intersection = true)
    names = collect(keys(dict))
    matches = if substrs isa Vector || substrs isa Tuple
        filter(names) do name
            if intersection
                all(str -> occursin(String(str), String(name)), substrs)
            else
                any(str -> occursin(String(str), String(name)), substrs)
            end
        end
    else
        filter(x -> occursin(String(substrs), String(Symbol(x))), names)
    end
    return matches, [dict[i] for i in matches]
end

"""
    param_symbols(::Type{T})

Return a `Tuple` of symbols describing the parameter names of a given blox type `T`. When given a non-type input, this function will call itself on the type of its input as a fallback method.
"""
function param_symbols end
param_symbols(x::T) where {T <: AbstractBlox} = param_symbols(T)
param_symbols(x) = error("The method `param_symbols` has not yet been implemented for the type $x.")

"""
    state_symbols(::Type{T})

Return a `Tuple` of symbols describing the state names of a given blox type `T`. When given a non-type input, this function will call itself on the type of its input as a fallback method.
"""
function state_symbols end
state_symbols(x::T) where {T <: AbstractBlox} = state_symbols(T)
state_symbols(x) = error("The method `state_symbols` has not yet been implemented for the type $x.")

"""
    input_symbols(::Type{T})

Return a `Tuple` of symbols describing the input names of a given blox type `T`. When given a non-type input, this function will call itself on the type of its input as a fallback method.
"""
function input_symbols end
input_symbols(x::T) where {T <: AbstractBlox} = input_symbols(T)
input_symbols(x) = error("The method `input_symbols` has not yet been implemented for the type $x.")

"""
    output_symbols(::Type{T})

Return a `Tuple` of symbols describing the output names of a given blox type `T`. When given a non-type input, this function will call itself on the type of its input as a fallback method.
"""
function output_symbols end
output_symbols(::Type{T}) where {T} = ()
output_symbols(x::T) where {T <: AbstractBlox} = output_symbols(T)

"""
    computed_property_symbols(::Type{T})

Return a `Tuple` of symbols describing the computed property names of a given blox type `T`. When given a non-type argument, this function will call itself on the type of its input as a fallback method.

See also `GraphDynamics.computed_properties`
"""
computed_property_symbols(::Type{T}) where {T} = propertynames(computed_properties(T))
computed_property_symbols(x::T) where {T <: AbstractBlox} = computed_property_symbols(T)

"""
    computed_property_with_inputs_symbols(::Type{T})

Return a `Tuple` of symbols describing the computed property names of a given blox type `T` which require inputs. When given a non-type argument, this function will call itself on the type of its input as a fallback method.

See also `GraphDynamics.computed_properties_with_inputs`
"""
computed_property_with_inputs_symbols(::Type{T}) where {T} = propertynames(computed_properties_with_inputs(T))
computed_property_with_inputs_symbols(x::T) where {T <: AbstractBlox} = computed_property_with_inputs_symbols(T)

function get_blox_by_name(g::GraphSystem, name::Union{String, Symbol})
    for b in nodes(g)
        nameof(b) == Symbol(name) && return b
    end
    return nothing
end

# GraphDynamics needs the full namespaced name with the outermost namespace
# to ensure it's properly detecting uniqueness
GraphDynamics.get_name(blox::AbstractBlox) = full_namespaced_nameof(blox)
