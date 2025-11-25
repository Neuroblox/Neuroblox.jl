"""
    LIFExciCircuit(; name, N_neurons, namespace, kwargs...)

Create a fully-connected circuit composed of [`LIFExciNeurons`](@ref), where the number of neurons in the circuit is `N_neurons`.

The keyword arguments of this constructor are propagated to [`LIFExciNeuron`](@ref).
"""
struct LIFExciCircuit <: AbstractComposite
    namespace
    parts
    system
    connector
    kwargs

    function LIFExciCircuit(;
        name, 
        N_neurons,
        namespace=nothing,
        g_L = 25 * 1e-6, # mS
        V_L = -70, # mV
        V_E = 0, # mV
        V_I = -70, # mV
        θ = -50, # mV
        V_reset = -55, # mV
        C = 0.5 * 1e-3, # mS / kHz 
        τ_AMPA = 2, # ms
        τ_GABA = 5, # ms
        τ_NMDA_decay = 100, # ms
        τ_NMDA_rise = 2, # ms
        t_refract = 2, # ms
        α = 0.5, # ms⁻¹
        g_AMPA = 0.05 * 1e-6, # mS
        g_AMPA_ext = 2.1 * 1e-6, # mS
        g_GABA = 1.3 * 1e-6, # mS
        g_NMDA = 0.165 * 1e-6, # mS  
        Mg = 1, # mM
        exci_scaling_factor = 1,
        inh_scaling_factor = 1,
        skip_system_creation=false,
        kwargs...
        )

        neurons = map(Base.OneTo(N_neurons)) do i
            LIFExciNeuron(;
                name = Symbol("neuron$i"),
                namespace = namespaced_name(namespace, name),
                g_L,
                V_L,
                V_E,
                V_I,
                θ,
                V_reset,
                C,
                τ_AMPA,
                τ_GABA,
                τ_NMDA_decay,
                τ_NMDA_rise,
                t_refract,
                α,
                g_AMPA,
                g_AMPA_ext,
                g_GABA,
                g_NMDA,
                Mg,
                exci_scaling_factor,
                inh_scaling_factor
            )
        end

        g = MetaDiGraph()
        add_blox!.(Ref(g), neurons)

        for i in eachindex(neurons)
            for j in eachindex(neurons)
                add_edge!(g, i, j, Dict(kwargs))
            end
        end

        bc = connectors_from_graph(g)

        if skip_system_creation
            sys = nothing
        else
            sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(neurons; name)
        end
        new(namespace, neurons, sys, bc, kwargs)
    end
end

"""
    LIFInhCircuit(; name, N_neurons, namespace, kwargs...)

Create a fully-connected circuit composed of [`LIFInhNeurons`](@ref), where the number of neurons in the circuit is `N_neurons`.

The keyword arguments of this constructor are propagated to [`LIFInhNeuron`](@ref).
"""
struct LIFInhCircuit <: AbstractComposite
    namespace
    parts
    system
    connector
    kwargs

    function LIFInhCircuit(;
        name, 
        N_neurons,
        namespace=nothing,
        g_L = 20 * 1e-6, # mS
        V_L = -70, # mV
        V_E = 0, # mV
        V_I = -70, # mV
        θ = -50, # mV
        V_reset = -55, # mV
        C = 0.2 * 1e-3, # mS / kHz 
        τ_AMPA = 2, # ms
        τ_GABA = 5, # ms
        t_refract = 1, # ms
        α = 0.5, # ms⁻¹
        g_AMPA = 0.04 * 1e-6, # mS
        g_AMPA_ext = 1.62 * 1e-6, # mS
        g_GABA = 1 * 1e-6, # mS
        g_NMDA = 0.13 * 1e-6, # mS 
        Mg = 1, # mM 
        exci_scaling_factor = 1,
        inh_scaling_factor = 1,
        skip_system_creation = false,
        kwargs...
        )

        neurons = map(Base.OneTo(N_neurons)) do i
            LIFInhNeuron(;
                name = Symbol("neuron$i"),
                namespace = namespaced_name(namespace, name),
                g_L,
                V_L,
                V_E,
                V_I,
                θ,
                V_reset,
                C,
                τ_AMPA,
                τ_GABA,
                t_refract,
                α,
                g_AMPA,
                g_AMPA_ext,
                g_GABA,
                g_NMDA,
                Mg,
                exci_scaling_factor,
                inh_scaling_factor
            )
        end

        g = MetaDiGraph()
        add_blox!.(Ref(g), neurons)

        for i in eachindex(neurons)
            for j in eachindex(neurons)
                add_edge!(g, i, j, Dict(kwargs))
            end
        end

        bc = connectors_from_graph(g)

        if skip_system_creation
            sys = nothing
        else
            sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(neurons; name)
        end
        
        new(namespace, neurons, sys, bc, kwargs)
    end
end
