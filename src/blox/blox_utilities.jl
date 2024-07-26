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

function get_exci_neurons(g::MetaDiGraph)
    mapreduce(x -> get_exci_neurons(x), vcat, get_blox(g))
end

function get_exci_neurons(b::AbstractComponent)
    mapreduce(x -> get_exci_neurons(x), vcat, b.parts)
end

function get_inh_neurons(b::AbstractComponent)
    mapreduce(x -> get_inh_neurons(x), vcat, b.parts)
end

function get_discrete_parts(b::AbstractComponent)
    mapreduce(x -> get_discrete_parts(x), vcat, b.parts)
end

function get_exci_neurons(b::CompositeBlox)
    mapreduce(x -> get_exci_neurons(x), vcat, b.parts)
end

function get_inh_neurons(b::CompositeBlox)
    mapreduce(x -> get_inh_neurons(x), vcat, b.parts)
end

function get_discrete_parts(b::CompositeBlox)
    mapreduce(x -> get_discrete_parts(x), vcat, b.parts)
end

get_exci_neurons(n::AbstractExciNeuronBlox) = n
get_exci_neurons(n) = []

get_inh_neurons(n::AbstractInhNeuronBlox) = n
get_inh_neurons(n) = []

get_sys(blox) = blox.odesystem
get_sys(sys::AbstractODESystem) = sys

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
function get_input_equations(blox)
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

get_continuous_callbacks(bc::BloxConnector) = bc.continuous_callbacks
get_continuous_callbacks(blox::Union{CompositeBlox, AbstractComponent}) = (get_continuous_callbacks ∘ get_connector)(blox)
get_continuous_callbacks(blox) = []

get_weight_learning_rules(bc::BloxConnector) = bc.learning_rules
get_weight_learning_rules(blox::Union{CompositeBlox, AbstractComponent}) = (get_weight_learning_rules ∘ get_connector)(blox)
get_weight_learning_rules(blox) = Dict{Num, AbstractLearningRule}()

get_blox_parts(blox::Union{CompositeBlox, AbstractComponent}) = blox.parts

function get_weight(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :weight)
        return kwargs[:weight]
    else
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_gap_weight(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :gap_weight)
        return kwargs[:gap_weight]
    else
        error("Gap junction weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_weightmatrix(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :weightmatrix)
        return kwargs[:weightmatrix]
    else
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_delay(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :delay)
        return kwargs[:delay]
    else
#        @warn "Delay constant from $name_blox1 to $name_blox2 is not specified. It is assumed that there is no delay."
        return 0
    end
end

function get_density(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :density)
        return kwargs[:density]
    else 
        error("Connection density from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_sta(kwargs, name_blox1, name_blox2)
    haskey(kwargs, :sta) ? kwargs[:sta] : false    
end

function get_gap(kwargs, name_blox1, name_blox2)
    haskey(kwargs, :gap) ? kwargs[:gap] : false    
end

function get_event_time(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :t_event)
        return kwargs[:t_event]
    else 
        error("Time for the event that affects the connection from $name_blox1 to $name_blox2 is not specified.")
    end
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
    spikes, _ = argmaxima(x)
    peakproms!(spikes, x; minprom, maxheight)
    peakheights!(spikes, xx[spikes]; minheight, maxheight)

    return spikes
end

function count_spikes(x::AbstractVector{T}; minprom=zero(T), maxprom=nothing, minheight=zero(T), maxheight=nothing) where {T}
    spikes = find_spikes(x; minprom, maxprom, minheight, maxheight)
    
    return length(spikes)
end

"""
    function get_dynamic_states(sys)
    
    Function extracts states from the system that are dynamic variables, 
    get also indices of external inputs (u(t)) and measurements (like bold(t))
    Arguments:
    - `sys`: MTK system

    Returns:
    - `sts`  : states of the system that are neither external inputs nor measurements, i.e. these are the dynamic states
    - `idx_u`: indices of states that represent external inputs
    - `idx_m`: indices of states that represent measurements
"""
function get_dynamic_states(sys)
    sts = []
    idx = []
    for (i, s) in enumerate(unknowns(sys))
        if !((getdescription(s) == "ext_input") || (getdescription(s) == "measurement"))
            push!(sts, s)
            push!(idx, i)
        end
    end
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
    return idx
end

function get_idx_tagged_vars(sys, tag)
    idx = Int[]
    for (i, s) in enumerate(unknowns(sys))
        if (getdescription(s) == tag)
            push!(idx, i)
        end
    end
    return idx
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
    if haskey(kwargs, :connection_rule)
        cr = kwargs[:connection_rule]
    else
        name_blox1 = nameof(bloxout)
        name_blox1 = nameof(bloxin)
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
