has_t_block_event(::Type{<:AbstractDiscrete}) = true
is_t_block_event_time(::Type{<:AbstractDiscrete}, key, t) = key == :t_block_early
t_block_event_requires_inputs(::Type{<:AbstractDiscrete}) = true
function apply_t_block_event!(_, vparams, s::Subsystem{<:AbstractDiscrete}, (;jcn), _)
    params = get_params(s)
    vparams[] = @reset params.jcn_t_block = jcn
end

macro recursive_getdefaults(sys, names::Symbol...)
    gets = Expr(:tuple, (:($recursive_getdefault($getproperty($sys, $(QuoteNode(name)))))
                         for name ∈ names)...)
    esc(:(($(names...),) = $gets))
end

GraphDynamics.get_name(blox::AbstractDiscrete) = namespaced_nameof(blox)
