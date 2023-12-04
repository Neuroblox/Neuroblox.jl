# function progress_scope(params; lvl=0)
#     para_list = []
#     for p in params
#         pp = ModelingToolkit.unwrap(p)
#         if ModelingToolkit.hasdefault(pp)
#             d = ModelingToolkit.getdefault(pp)
#             if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
#                 if lvl==0
#                     pp = ParentScope(pp)
#                 else
#                     pp = DelayParentScope(pp,lvl)
#                 end
#             end
#         end
#         push!(para_list,ModelingToolkit.wrap(pp))
#     end
#     return para_list
# end

"""
This function progresses the scope of parameters and leaves floating point values untouched
"""
function progress_scope(args...)
    paramlist = []
    for p in args
        if p isa Num
            p = ParentScope(p)
            # pp = ModelingToolkit.unwrap(p)
            # if ModelingToolkit.hasdefault(pp)
            #     d = ModelingToolkit.getdefault(pp)
            #     if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
            #         pp = ParentScope(pp)
            #     end
            # end
            # push!(para_list,ModelingToolkit.wrap(pp))
            push!(paramlist, p)
        else
            push!(paramlist, p)
        end
    end
    return paramlist
end

"""
    This function compiles already existing parameters with floats after making them parameters.
    Keyword arguments are used because parameter definition requires names, not just values
"""
function compileparameterlist(;kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Union{Float64, Int}  # note that Num is also subtype of Real. Thus union of types seems to be the solution.
            paramlist = vcat(paramlist, @parameters $kw = v [tunable=true])
        else
            paramlist = vcat(paramlist, v)
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
    ODESystem(
        equations(sys), 
        independent_variable(sys), 
        states(sys), 
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
function input_equations(blox)
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

input_equations(blox::AbstractComponent) = blox.connector.eqs
input_equations(blox::CompositeBlox) = blox.connector.eqs
input_equations(::ImageStimulus) = []

weight_parameters(blox) = Num[]
weight_parameters(blox::AbstractComponent) = blox.connector.weights #I think this is the fix?
weight_parameters(blox::CompositeBlox) = blox.connector.weights #I think this is the fix?

delay_parameters(blox) = Num[]
delay_parameters(blox::AbstractComponent) = blox.connector.delays
delay_parameters(blox::CompositeBlox) = blox.connector.delays

event_callbacks(blox) = []
event_callbacks(blox::AbstractComponent) = blox.connector.events
event_callbacks(blox::CompositeBlox) = blox.connector.events

weight_learning_rules(blox) = Dict{Num, AbstractLearningRule}()
weight_learning_rules(bc::BloxConnector) = bc.learning_rules
weight_learning_rules(blox::AbstractComponent) = weight_learning_rules(blox.connector)
weight_learning_rules(blox::CompositeBlox) = weight_learning_rules(blox.connector)

function get_weight(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :weight)
        return kwargs[:weight]
    else
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_delay(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :delay)
        return kwargs[:delay]
    else
        @warn "Delay constant from $name_blox1 to $name_blox2 is not specified. It is assumed that there is no delay."
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

function get_event_time(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :t_event)
        return kwargs[:t_event]
    else 
        error("Time for the event that affects the connection from $name_blox1 to $name_blox2 is not specified.")
    end
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

function get_hemodynamic_observers(sys_from_graph, nr)
    obs_idx = Dict([k => [] for k in 1:nr])
    obs_states = Dict([k => [] for k in 1:nr])
    for (i, s) in enumerate(states(sys_from_graph))
        if isequal(getdescription(s), "hemodynamic_observer")
            regionidx = parse(Int64, split(string(s), "₊")[1][end])
            push!(obs_idx[regionidx], i)
            push!(obs_states[regionidx], s)
        end
    end
    return (obs_idx, obs_states)
end

function addnontunableparams(param, model)
    newparam = []
    k = 0
    for p in parameters(model)
        if istunable(p)
            k += 1
            push!(newparam, param[k])
        else
            push!(newparam, Symbolics.getdefaultval(p))
        end
    end
    append!(newparam, param[k+1:end])
    return newparam
end