function Base.getproperty(b::Union{AbstractNeuronBlox, NeuralMassBlox}, name::Symbol)
    # TO DO : Some of the fields below besides `system` and `namespace` 
    # are redundant and we should clean them up. 
    if (name === :system) || (name === :namespace) || (name === :params)
        return getfield(b, name)
    else
        return Base.getproperty(Neuroblox.get_namespaced_sys(b), name)
    end
end

"""
    function paramscoping(;tunable=true, kwargs...)
    
    Scope arguments that are already a symbolic model parameter thereby keep the correct namespace 
    and make those that are not yet symbolic a symbol.
    Keyword arguments are used, because parameter definition require names, not just values.
"""
function paramscoping(;tunable=true, kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Num
            paramlist = vcat(paramlist, ParentScope(v))
        else
            paramlist = vcat(paramlist, @parameters $kw = v [tunable=tunable])
        end
    end
    return paramlist
end

function changetune(model, parlist)
    parstochange = keys(parlist)
    p_new = map(parameters(model)) do p
        p in parstochange ? setmetadata(p, ModelingToolkit.VariableTunable, parlist[p]) : p
    end
    System(equations(model), ModelingToolkit.get_iv(model), unknowns(model), p_new; name=model.name)
end

get_HH_exci_neurons(n::HHNeuronExciBlox) = [n]
get_HH_exci_neurons(n) = []

function get_HH_exci_neurons(g::MetaDiGraph)
    mapreduce(x -> get_HH_exci_neurons(x), vcat, get_bloxs(g))
end

function get_HH_exci_neurons(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_HH_exci_neurons(x), vcat, b.parts)
end

get_exci_neurons(n::AbstractExciNeuronBlox) = [n]
get_exci_neurons(n) = []

function get_exci_neurons(g::MetaDiGraph)
    mapreduce(x -> get_exci_neurons(x), vcat, get_bloxs(g))
end

function get_exci_neurons(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_exci_neurons(x), vcat, b.parts)
end

get_inh_neurons(n::AbstractInhNeuronBlox) = [n]
get_inh_neurons(n) = []

function get_inh_neurons(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_inh_neurons(x), vcat, b.parts)
end

get_neurons(n::AbstractNeuronBlox) = [n]
get_neurons(n) = []

function get_neurons(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_neurons(x), vcat, b.parts)
end

function get_neurons(vn::AbstractVector{<:AbstractBlox})
    mapreduce(x -> get_neurons(x), vcat, vn)
end

get_parts(blox::CompositeBlox) = blox.parts
get_parts(blox::Union{AbstractBlox, ObserverBlox}) = blox

get_components(blox::CompositeBlox) = mapreduce(get_components, vcat, get_parts(blox))
get_components(blox::Vector{<:AbstractBlox}) = mapreduce(get_components, vcat, blox)
get_components(blox) = [blox]

get_dynamics_components(blox::AbstractDiscrete) = []
get_dynamics_components(blox::Union{NeuralMassBlox, AbstractNeuronBlox}) = [blox]
get_dynamics_components(blox::CompositeBlox) = mapreduce(get_dynamics_components, vcat, get_parts(blox))
get_dynamics_components(blox::Vector{<:AbstractBlox}) = mapreduce(get_dynamics_components, vcat, blox)

get_neuron_color(n::AbstractExciNeuronBlox) = "blue"
get_neuron_color(n::AbstractInhNeuronBlox) = "red"
get_neuron_color(n::AbstractNeuronBlox) = "black"

get_neuron_color(n::Union{CompositeBlox, Vector{<:AbstractBlox}}) = map(get_neuron_color, get_neurons(n))

function get_discrete_parts(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_discrete_parts(x), vcat, b.parts)
end

get_system(blox) = blox.system
get_system(sys::AbstractODESystem) = sys
get_system(stim::AbstractSpikeSource) = System(Equation[], t, [], []; name=stim.name)

function system(blox::CompositeBlox; simplify=true)
    sys = get_system(blox)
    eqs = get_input_equations(blox; namespaced=false)

    csys = System(vcat(equations(sys), eqs), t, unknowns(sys), parameters(sys); name = nameof(sys))

    return simplify ? structural_simplify(csys) : csys
end


function system(blox::AbstractBlox; simplify=true, kwargs...)
    sys = get_system(blox)
    eqs = get_input_equations(blox; namespaced=true)
    csys =  compose(System(eqs, t, [], []; name=namespaced_nameof(blox), kwargs...), sys)

    return simplify ? structural_simplify(csys) : csys
end

function get_namespaced_sys(blox)
    sys = get_system(blox)

    System(
        equations(sys), 
        only(independent_variables(sys)), 
        unknowns(sys), 
        parameters(sys); 
        name = namespaced_nameof(blox),
        discrete_events = discrete_events(sys)
    ) 
end

get_namespaced_sys(sys::AbstractODESystem) = sys

nameof(blox) = (nameof ∘ get_system)(blox)
nameof(blox::AbstractActionSelection) = blox.name

namespaceof(blox) = blox.namespace

namespaced_nameof(blox) = namespaced_name(inner_namespaceof(blox), nameof(blox))

"""
    Returns the complete namespace EXCLUDING the outermost (highest) level.
    This is useful for manually preparing equations (e.g. connections, see Connector),
    that will later be composed and will automatically get the outermost namespace.
""" 
function inner_namespaceof(blox)
    parts = split((string ∘ namespaceof)(blox), '₊')
    if length(parts) == 1
        return nothing
    else
        return join(parts[2:end], '₊')
    end
end

namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(::Nothing, name) = Symbol(name)

function find_eq(eqs::Union{AbstractVector{<:Equation}, Equation}, lhs)
    findfirst(eqs) do eq
        lhs_vars = get_variables(eq.lhs)
        length(lhs_vars) == 1 && isequal(only(lhs_vars), lhs)
    end
end

function ModelingToolkit.outputs(blox::AbstractBlox; namespaced=false)
    sys = get_namespaced_sys(blox)
    
    # Wrap in Num for convenience when checking `isa Num` to resolve delay or no delay connection.
    return namespaced ? Num.(namespace_expr.(ModelingToolkit.outputs(sys), Ref(sys))) : Num.(ModelingToolkit.outputs(sys))
end 

function ModelingToolkit.inputs(blox::AbstractBlox; namespaced=false)
    sys = get_namespaced_sys(blox)
    
    # Wrap in Num for convenience when checking `isa Num` to resolve delay or no delay connection.
    return namespaced ? Num.(namespace_expr.(ModelingToolkit.inputs(sys), Ref(sys))) : Num.(ModelingToolkit.inputs(sys))
end 

ModelingToolkit.equations(blox::AbstractBlox) = ModelingToolkit.equations(get_namespaced_sys(blox))

ModelingToolkit.discrete_events(blox::AbstractBlox) = ModelingToolkit.discrete_events(get_namespaced_sys(blox))

ModelingToolkit.unknowns(blox::AbstractBlox) = ModelingToolkit.unknowns(get_namespaced_sys(blox))

ModelingToolkit.parameters(blox::AbstractBlox) = ModelingToolkit.parameters(get_namespaced_sys(blox))

"""
    Returns the equations for all input variables of a system, 
    assuming they have a form like : `sys.input_variable ~ ...`
    so only the input appears on the LHS.

    Input equations are namespaced by the inner namespace of blox
    and then they are returned. This way during system `compose` downstream,
    the higher-level namespaces will be added to them.

    If blox isa AbstractComponent, it is assumed that it contains a `connector` field,
    which holds a `Connector` object with all relevant connections 
    from lower levels and this level.
"""
function get_input_equations(blox::Union{AbstractBlox, ObserverBlox}; namespaced=true)
    sys = get_system(blox)
    sys_eqs = equations(sys)

    inps = inputs(sys)
    filter!(inp -> isnothing(find_eq(sys_eqs, inp)), inps)

    if !isempty(inps)
        eqs = if namespaced
            map(inps) do inp
                namespace_equation(
                    inp ~ 0, 
                    sys,
                    namespaced_name(inner_namespaceof(blox), nameof(blox))
                ) 
            end
        else
            map(inps) do inp
                inp ~ 0
            end
        end

        return eqs
    else
        return Equation[]
    end
end

get_input_equations(blox) = []

get_connectors(blox::Union{CompositeBlox, Agent}) = blox.connector
get_connectors(blox) = [Connector(namespaced_nameof(blox), namespaced_nameof(blox))]

get_connector(blox::Union{CompositeBlox, Agent}) = reduce(merge!, get_connectors(blox))
get_connector(blox) = Connector(namespaced_nameof(blox), namespaced_nameof(blox))

function get_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :weight) do
        @info "Connection weight from $name_blox1 to $name_blox2 is not specified. Assuming weight=1" 
        return 1.0
    end
end

function get_gap_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :gap_weight) do
        error("Gap junction weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

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

function get_density(kwargs, name_blox1, name_blox2)
    get(kwargs, :density) do 
        error("Connection density from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_sta(kwargs, name_blox1, name_blox2)
    get(kwargs, :sta, false)
end

function get_gap(kwargs, name_blox1, name_blox2)
    get(kwargs, :gap, false)    
end

function get_event_time(kwargs, name_blox1, name_blox2)
    get(kwargs, :t_event) do
        error("Time for the event that affects the connection from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_connection_matrix(kwargs, name_out, name_in, N_out, N_in)
    sz = (N_out, N_in)
    connection_matrix = get(kwargs, :connection_matrix) do
        density = get_density(kwargs, name_out, name_in)
        dist = Bernoulli(density)
        rng = get(kwargs, :rng, Random.default_rng())
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

function get_weights(agent::Agent, blox_out, blox_in)
    ps = parameters(agent.system)
    pv = agent.problem.p
    map_idxs = Int.(ModelingToolkit.varmap_to_vars([ps[i] => i for i in eachindex(ps)], ps))

    name_out = String(namespaced_nameof(blox_out))
    name_in = String(namespaced_nameof(blox_in))

    idxs_weight = findall(ps) do p
        n = String(Symbol(p))
        r = Regex("w.*$(name_out).*$(name_in)")
        occursin(r, n)
    end
    
    return pv[map_idxs[idxs_weight]]
end

"""
    function get_dynamic_states(sys)
    
    Function extracts states from the system that are dynamic variables, 
    get also indices of external inputs (u(t)) and measurements (like bold(t))
    Arguments:
    - `sys`: MTK system

    Returns:
    - `sts`: states/unknowns of the system that are neither external inputs nor measurements, i.e. these are the dynamic states
    - `idx`: indices of these states
"""
function get_dynamic_states(sys)
    itr = Iterators.filter(enumerate(unknowns(sys))) do (_, s)
        !((getdescription(s) == "ext_input") || (getdescription(s) == "measurement"))
    end
    sts = map(x -> x[2], itr)
    idx = map(x -> x[1], itr)
    return sts, idx
end

function get_eqidx_tagged_vars(sys, tag)
    idx = Int[]
    vars = []
    eqs = equations(sys)
    for s in unknowns(sys)
        if getdescription(s) == tag
            push!(vars, s)
        end
    end

    for v in vars
        for (i, e) in enumerate(eqs)
            for s in Symbolics.get_variables(e)
                if string(s) == string(v)
                    push!(idx, i)
                end
            end
        end
    end
    return idx, vars
end

function get_idx_tagged_vars(sys, tag)
    idx = Int[]
    for (i, s) in enumerate(unknowns(sys))
        if (getdescription(s) == tag)
            push!(idx, i)
        end
    end
    return idx
end

"""
    function addnontunableparams(param, model)
    
    Function adds parameters of a model that were not marked as tunable to a list of tunable parameters
    and respects the MTK ordering of parameters.

    Arguments:
    - `paramlist`: parameters of an MTK system that were tagged as tunable
    - `sys`: MTK system

    Returns:
    - `completeparamlist`: complete parameter list of a system, including those that were not tagged as tunable
"""
function addnontunableparams(paramlist, sys)
    completeparamlist = []
    k = 0
  
    for p in parameters(sys)
        if istunable(p)
            k += 1
            push!(completeparamlist, paramlist[k])
        else
            push!(completeparamlist, Symbolics.getdefaultval(p))
        end
    end
    append!(completeparamlist, paramlist[k+1:end])
    return completeparamlist
end

function get_connection_rule(kwargs, bloxout, bloxin, w)
    cr = get(kwargs, :connection_rule) do
        name_blox1 = nameof(bloxout)
        name_blox2 = nameof(bloxin)
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

to_double_vector(v::AbstractVector{<:AbstractVector}) = v
to_double_vector(v::AbstractVector) = [v]
to_double_vector(v) = [[v]]

nanmean(x) = mean(filter(!isnan,x))

function replace_refractory!(V, blox::Union{LIFExciNeuron, LIFInhNeuron}, sol::SciMLBase.AbstractSolution)
    namespaced_name = namespaced_nameof(blox)
    reset_param_name = Symbol(namespaced_name, "₊V_reset")

    get_reset = getp(sol, reset_param_name)
    reset_value = get_reset(sol)

    V[V .== reset_value] .= NaN

    return V
end

function replace_refractory!(V, blox::CompositeBlox, sol::SciMLBase.AbstractSolution)
    neurons = get_neurons(blox)

    for (i, n) in enumerate(neurons)
        V[:, i] = replace_refractory!(V[:,i], n, sol)
    end
end

replace_refractory!(V, blox, sol::SciMLBase.AbstractSolution) = V

function find_spikes(x::AbstractVector{T}; threshold=zero(T)) where {T}    
    spike_idxs = argmaxima(x)
    peakheights!(spike_idxs, x[spike_idxs]; minheight = threshold)
    
    spikes = sparsevec(spike_idxs, ones(length(spike_idxs)), length(x))

    return spikes
end


function count_spikes(x::AbstractVector{T}; threshold=zero(T)) where {T}
    spikes = find_spikes(x; threshold)
    
    return nnz(spikes)
end

function detect_spikes(
    blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution; 
    threshold = nothing, tolerance = 1e-3, ts = nothing, kwargs...
)
    namespaced_name = namespaced_nameof(blox)

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
    blox::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}}, sol::SciMLBase.AbstractSolution;
    threshold = nothing, ts=nothing, scheduler=:serial, kwargs...
)

    neurons = get_neurons(blox)

    S = tmapreduce(sparse_hcat, neurons; scheduler, kwargs...) do neuron
        detect_spikes(neuron, sol; threshold, ts)
    end

    return S
end

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
    transient = 0, win_size = last(sols[1].t) - transient, overlap = 0,
    threshold = nothing, scheduler=:serial, kwargs...)

    firing_rates = map(sols) do sol
        firing_rate(blox, sol; transient, win_size, overlap, threshold, scheduler, kwargs...)
    end

    mean_fr = mean(firing_rates)
    std_fr = std(firing_rates)

    return mean_fr, std_fr
end

function state_timeseries(blox, sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
                          
    namespaced_name = namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")

    if isnothing(ts)
        return sol[state_name]
    else
        return Array(sol(ts, idxs = state_name))
    end
end

function state_timeseries(cb::Union{CompositeBlox, AbstractVector{<:AbstractBlox}},
                          sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
    
    neurons = get_dynamics_components(cb)
    state_names = map(neuron -> Symbol(namespaced_nameof(neuron), "₊", state), neurons)

    if isnothing(ts)
        s = stack(sol[state_names], dims=1)
    else
        s = transpose(Array(sol(ts; idxs=state_names)))
    end

    return s
end

function meanfield_timeseries(cb::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}},
                              sol::SciMLBase.AbstractSolution, state::String; ts=nothing)
                              
    s = state_timeseries(cb, sol, state; ts)

    return vec(mapslices(nanmean, s; dims = 2))
end

voltage_timeseries(blox, sol::SciMLBase.AbstractSolution; ts=nothing) = 
    state_timeseries(blox, sol, "V"; ts)

function voltage_timeseries(cb::Union{CompositeBlox, AbstractVector{<:AbstractBlox}},
                            sol::SciMLBase.AbstractSolution; ts=nothing)
    return state_timeseries(cb, sol, "V"; ts)
end

function meanfield_timeseries(cb::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}},
                              sol::SciMLBase.AbstractSolution; ts=nothing)
    V = voltage_timeseries(cb, sol; ts)
    replace_refractory!(V, cb, sol)

    return vec(mapslices(nanmean, V; dims = 2))
end

function powerspectrum(cb::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}},
                       sol::SciMLBase.AbstractSolution, state::String; sampling_rate=nothing,
                       method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    s = meanfield_timeseries(cb, sol, state; ts = t_sampled)

    return method(s, fs=sampling_freq, window=window)
end

function powerspectrum(cb::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}},
                       sol::SciMLBase.AbstractSolution; sampling_rate=nothing,
                       method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    V = voltage_timeseries(cb, sol; ts = t_sampled)
    replace_refractory!(V, cb, sol)

    return method(vec(mapslices(nanmean, V; dims = 2)), fs = sampling_freq, window=window)
end

function powerspectrum(blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution, state::String;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    namespaced_name = namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    data = isnothing(t_sampled) ? sol[state_name] : Array(sol(t_sampled, idxs = state_name))

    return method(data, fs = sampling_freq, window=window)
end

function powerspectrum(cb::Union{CompositeBlox, AbstractVector{<:AbstractNeuronBlox}},
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

function powerspectrum(cb::NeuralMassBlox, sol::SciMLBase.AbstractSolution, state::String;
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

function narrowtype_union(d::Dict)
    types = unique(typeof.(values(d)))
    U = Union{types...}

    return U
end

function narrowtype(d::Dict)
    U = narrowtype_union(d)

    return Dict{Num, U}(d)
end
