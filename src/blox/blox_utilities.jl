function progress_scope(params; lvl=0)
    para_list = []
    for p in params
        pp = ModelingToolkit.unwrap(p)
        if ModelingToolkit.hasdefault(pp)
            d = ModelingToolkit.getdefault(pp)
            if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
                if lvl==0
                    pp = ParentScope(pp)
                else
                    pp = DelayParentScope(pp,lvl)
                end
            end
        end
        push!(para_list,ModelingToolkit.wrap(pp))
    end
    return para_list
end

"""
This function progresses the scope of parameters and leaves floating point values untouched
"""
function progress_scope(args...)
    paramlist = []
    for p in args
        if p isa Float64
            push!(paramlist, p)
        else
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
        if v isa Float64
            paramlist = vcat(paramlist, @parameters $kw = v)
        else
            paramlist = vcat(paramlist, v)
        end
    end
    return paramlist
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

get_exci_neurons(n::AbstractExciNeuronBlox) = n
get_exci_neurons(n) = []

get_inh_neurons(n::AbstractInhNeuronBlox) = n
get_inh_neurons(n) = []

get_discrete_parts(n::AbstractDiscrete) = n
get_discrete_parts(n) = []

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

input_equations(::ImageStimulus) = []

weight_parameters(blox) = Num[]
weight_parameters(blox::AbstractComponent) = blox.connector.weights #I think this is the fix?

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
