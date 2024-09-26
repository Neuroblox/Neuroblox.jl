"""
    function paramscoping(;kwargs...)
    
    Scope arguments that are already a symbolic model parameter thereby keep the correct namespace 
    and make those that are not yet symbolic a symbol.
    Keyword arguments are used, because parameter definition require names, not just values.
"""
function paramscoping(;kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Num
            paramlist = vcat(paramlist, ParentScope(v))
        else
            paramlist = vcat(paramlist, @parameters $kw = v [tunable=true])
        end
    end
    return paramlist
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


function get_discrete_parts(b::Union{AbstractComponent, CompositeBlox})
    mapreduce(x -> get_discrete_parts(x), vcat, b.parts)
end

get_sys(blox) = blox.odesystem
get_sys(sys::AbstractODESystem) = sys
get_sys(stim::PoissonSpikeTrain) = System(Equation[], t, [], []; name=stim.name)

function get_namespaced_sys(blox)
    sys = get_sys(blox)
    System(
        equations(sys), 
        only(independent_variables(sys)), 
        unknowns(sys), 
        parameters(sys); 
        name = namespaced_nameof(blox)
    ) 
end

get_namespaced_sys(sys::AbstractODESystem) = sys

nameof(blox) = (nameof ∘ get_sys)(blox)

namespaceof(blox) = blox.namespace

namespaced_nameof(blox) = namespaced_name(inner_namespaceof(blox), nameof(blox))

"""
    Returns the complete namespace EXCLUDING the outermost (highest) level.
    This is useful for manually preparing equations (e.g. connections, see BloxConnector),
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

function find_eq(eqs::AbstractVector{<:Equation}, lhs)
    findfirst(eqs) do eq
        lhs_vars = get_variables(eq.lhs)
        length(lhs_vars) == 1 && isequal(only(lhs_vars), lhs)
    end
end

"""
    Returns the equations for all input variables of a system, 
    assuming they have a form like : `sys.input_variable ~ ...`
    so only the input appears on the LHS.

    Input equations are namespaced by the inner namespace of blox
    and then they are returned. This way during system `compose` downstream,
    the higher-level namespaces will be added to them.

    If blox isa AbstractComponent, it is assumed that it contains a `connector` field,
    which holds a `BloxConnector` object with all relevant connections 
    from lower levels and this level.
"""
function get_input_equations(blox::Union{AbstractBlox, ObserverBlox})
    sys = get_sys(blox)
    inps = inputs(sys)
    sys_eqs = equations(sys)

    @variables t # needed for IV in namespace_equation

    eqs = map(inps) do inp
        idx = find_eq(sys_eqs, inp)
        if isnothing(idx)
            namespace_equation(
                inp ~ 0, 
                sys,
                namespaced_name(inner_namespaceof(blox), nameof(blox))
            ) 
        else
            namespace_equation(
                sys_eqs[idx],
                sys,
                namespaced_name(inner_namespaceof(blox), nameof(blox))
            )
        end
    end

    return eqs
end

get_connector(blox::Union{CompositeBlox, AbstractComponent}) = blox.connector

get_input_equations(bc::BloxConnector) = bc.eqs
get_input_equations(blox::Union{CompositeBlox, AbstractComponent}) = (get_input_equations ∘ get_connector)(blox)
get_input_equations(blox) = []

get_weight_parameters(bc::BloxConnector) = bc.weights
get_weight_parameters(blox::Union{CompositeBlox, AbstractComponent}) = (get_weight_parameters ∘ get_connector)(blox)
get_weight_parameters(blox) = Num[]

get_delay_parameters(bc::BloxConnector) = bc.delays
get_delay_parameters(blox::Union{CompositeBlox, AbstractComponent}) = (get_delay_parameters ∘ get_connector)(blox)
get_delay_parameters(blox) = Num[]

get_discrete_callbacks(bc::BloxConnector) = bc.discrete_callbacks
get_discrete_callbacks(blox::Union{CompositeBlox, AbstractComponent}) = (get_discrete_callbacks ∘ get_connector)(blox)
get_discrete_callbacks(blox) = []

get_spike_affect_states(bc::BloxConnector) = bc.spike_affect_states
get_spike_affect_states(blox::Union{CompositeBlox, AbstractComponent}) = (get_spike_affect_states ∘ get_connector)(blox)
get_spike_affect_states(blox) = Dict{Symbol, Vector{Num}}()

get_weight_learning_rules(bc::BloxConnector) = bc.learning_rules
get_weight_learning_rules(blox::Union{CompositeBlox, AbstractComponent}) = (get_weight_learning_rules ∘ get_connector)(blox)
get_weight_learning_rules(blox) = Dict{Num, AbstractLearningRule}()

get_blox_parts(blox::Union{CompositeBlox, AbstractComponent}) = blox.parts

function get_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :weight) do
        @warn "Connection weight from $name_blox1 to $name_blox2 is not specified. Assuming weight=1" 
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

function get_weights(agent::Agent, blox_out, blox_in)
    ps = parameters(agent.odesystem)
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

function find_spikes(x::AbstractVector{T}; minprom=zero(T), maxprom=nothing, minheight=zero(T), maxheight=nothing) where {T}
    spikes = argmaxima(x)
    peakproms!(spikes, x; minprom, maxprom)
    peakheights!(spikes, x[spikes]; minheight, maxheight)

    return spikes
end

function find_spikes(x::AbstractMatrix{T}; threshold=-50) where {T}
    spikes = spzeros(size(x))

    for i in 1:size(x, 2)
        pks, vals = findmaxima(x[:, i])
        idx = findall(vals .> threshold)
        spikes[pks[idx], i] .= 1
    end
    return SparseMatrixCSC(transpose(spikes))
end

function count_spikes(x::AbstractVector{T}; minprom=zero(T), maxprom=nothing, minheight=zero(T), maxheight=nothing) where {T}
    spikes = find_spikes(x; minprom, maxprom, minheight, maxheight)
    
    return length(spikes)
end

function detect_spikes(blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution; tolerance = 1e-3)
    namespaced_name = namespaced_nameof(blox)
    reset_param_name = Symbol(namespaced_name, "₊V_reset")
    threshold_param_name = Symbol(namespaced_name, "₊θ")

    reset = only(@parameters $(reset_param_name))
    thrs = only(@parameters $(threshold_param_name))

    get_reset = getp(sol, reset)
    reset_value = get_reset(sol)

    get_thrs = getp(sol, thrs)
    thrs_value = get_thrs(sol)

    V = voltage_timeseries(blox, sol)
    
    spikes = find_spikes(V; minheight = thrs_value - tolerance)

    return spikes
end

function detect_spikes(blox::CompositeBlox, sol::SciMLBase.AbstractSolution;
                       threshold = -50.0, ts=nothing)
    namespaced_name = namespaced_nameof(blox)

    if isnothing(threshold)
        threshold_param_name = Symbol(namespaced_name, "₊θ")
        thrs = only(@parameters $(threshold_param_name))
        get_thrs = getp(sol, threshold_param_name)
        thrs_value = get_thrs(sol)
    else
        thrs_value = threshold
    end

    V = voltage_timeseries(blox, sol, ts)
    spikes = find_spikes(V, threshold = thrs_value)

    return spikes
end

function mean_firing_rate(spikes::SparseMatrixCSC, sol; trim_transient = 0,
                 firing_rate_Δt = last(sol.t) - trim_transient,)

    tmax = last(sol.t) - trim_transient
    t = trim_transient:firing_rate_Δt:tmax
    @show t
    @show collect(t)
    tᵤ = unique(sol.t)
    counts = vec(sum(spikes, dims=1))

    rₘ = fill(NaN64, length(t) - 1)
    for i in 2:length(t)
        idx = intersect(findall(tᵤ .<= t[i]), findall(tᵤ .> t[i-1]))
        if ~isempty(idx)
            rₘ[i-1] = sum(counts[idx])
        end
    end

    # firing rate in spikes/s averaged over the population
    rₘ = rₘ*1000 ./ (size(spikes,1)*firing_rate_Δt)
    return t, rₘ
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
        @warn "Neuron connection rule from $name_blox1 to $name_blox2 is not specified. It is assumed that there is a basic weighted connection."
        cr = "basic"
    end

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

     # Logic based on connection rule type
     if isequal(cr, "basic")
        x = namespace_expr(bloxout.output, sys_out)
        rhs = x*w
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

function replace_refractory!(V, blox::Union{LIFExciNeuron, LIFInhNeuron}, sol::SciMLBase.AbstractSolution)
    namespaced_name = namespaced_nameof(blox)
    reset_param_name = Symbol(namespaced_name, "₊V_reset")
    p = only(@parameters $(reset_param_name))

    get_reset = getp(sol, p)
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

function state_timeseries(blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution,
                          state::String; ts=nothing)
                          
    namespaced_name = namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")
    s = only(@variables $(state_name)(t))

    if isnothing(ts)
        return sol[s]
    else
        return Array(sol(ts, idxs = state_name))
    end
end

function state_timeseries(cb::CompositeBlox, sol::SciMLBase.AbstractSolution, state::String; ts=nothing)

    return mapreduce(hcat, get_neurons(cb)) do neuron
        state_timeseries(neuron, sol, state; ts)
    end
end

function meanfield_timeseries(cb::CompositeBlox, sol::SciMLBase.AbstractSolution,
                              state::String; ts=nothing)
                              
    s = state_timeseries(cb, sol, state; ts)

    return vec(mapslices(nanmean, s; dims = 2))
end

voltage_timeseries(blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution; ts=nothing) = 
    state_timeseries(blox, sol, "V"; ts)

function voltage_timeseries(cb::Union{CompositeBlox, AbstractVector{<:AbstractBlox}}, sol::SciMLBase.AbstractSolution; ts=nothing)

    return mapreduce(hcat, get_neurons(cb)) do neuron
        voltage_timeseries(neuron, sol; ts)
    end
end

function meanfield_timeseries(cb::CompositeBlox, sol::SciMLBase.AbstractSolution; ts=nothing)
    V = voltage_timeseries(cb, sol; ts)
    replace_refractory!(V, cb, sol)

    return vec(mapslices(nanmean, V; dims = 2))
end

function powerspectrum(cb::CompositeBlox, sol::SciMLBase.AbstractSolution, state::String;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    s = meanfield_timeseries(cb, sol, state; ts = t_sampled)

    return method(s, fs=sampling_freq, window=window)
end

function powerspectrum(cb::CompositeBlox, sol::SciMLBase.AbstractSolution;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    V = voltage_timeseries(cb, sol; ts = t_sampled)
    replace_refractory!(V, cb, sol)

    return method(vec(mapslices(nanmean, V; dims = 2)), fs = sampling_freq, window=window)
end

function powerspectrum(blox::AbstractNeuronBlox, sol::SciMLBase.AbstractSolution, state::String;
                       sampling_rate=nothing, method=periodogram, window=nothing)

    namespaced_name = namespaced_nameof(blox)
    state_name = Symbol(namespaced_name, "₊$(state)")
    s = only(@variables $(state_name)(t))

    t_sampled, sampling_freq = get_sampling_info(sol; sampling_rate=sampling_rate)
    data = isnothing(t_sampled) ? sol[s] : Array(sol(t_sampled, idxs = state_name))

    return method(data, fs = sampling_freq, window=window)
end

function get_sampling_info(sol::SciMLBase.AbstractSolution; sampling_rate=nothing)
    t_raw = unique(sol.t)
    dt = diff(t_raw)
    dt_std = std(dt)
    first_diff = dt[1]

    # check if the solution was saved at regular time steps
    if !isapprox(dt_std, 0, atol=1e-10 * first_diff)
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