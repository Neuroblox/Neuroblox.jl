function replace_refractory!(V, blox::Union{LIFExciNeuron, LIFInhNeuron}, sol::SciMLBase.AbstractSolution)
    namespaced_name = full_namespaced_nameof(blox)
    reset_param_name = Symbol(namespaced_name, "₊V_reset")

    get_reset = getp(sol, reset_param_name)
    reset_value = get_reset(sol)

    V[V .== reset_value] .= NaN

    return V
end
