struct CorticalBlox <: CompositeBlox
    namespace
    parts
    system
    connector
    kwargs

    function CorticalBlox(;
        name, 
        N_wta=20,
        namespace=nothing,
        N_exci=5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=4.0,
        G_syn_ff_inhib=3.5,
        freq=0.0,
        phase=0.0,
        I_bg_ar=0,
        τ_exci=5,
        τ_inhib=70,
        kwargs...
            )
        
        wtas = map(1:N_wta) do i
            if I_bg_ar isa Array
                I_bg = I_bg_ar[i]
            else
                I_bg = I_bg_ar
            end
            WinnerTakeAllBlox(;
                name=Symbol("wta$i"),
                namespace=namespaced_name(namespace, name),
                N_exci,
                E_syn_exci,
                E_syn_inhib,
                G_syn_exci,
                G_syn_inhib,
                I_bg = I_bg,
                freq,
                phase,
                τ_exci,
                τ_inhib    
            )
        end
        n_ff_inh = HHNeuronInhibBlox(
            name = "ff_inh",
            namespace = namespaced_name(namespace, name),
            E_syn = E_syn_inhib,
            G_syn = G_syn_ff_inhib,
            τ = τ_inhib
        )

        g = MetaDiGraph()
        add_blox!.(Ref(g), vcat(wtas, n_ff_inh))
        for i in 1:N_wta
            for j in 1:N_wta
                if j != i
                    # users can supply a matrix of connection matrices.
                    # connection_matrices[i,j][k, l] determines if neuron k from wta i is connected to
                    # neuron l from wta j.
                    if haskey(kwargs, :connection_matrices)
                        kwargs_ij = merge(kwargs, Dict(:connection_matrix => kwargs[:connection_matrices][i, j]))
                    else
                        kwargs_ij = Dict(kwargs)
                    end
                    add_edge!(g, i, j, kwargs_ij)
                end
            end
            add_edge!(g, N_wta+1, i, Dict(:weight => 1))
        end
        
        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the Connector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(vcat(wtas, n_ff_inh); name)

        new(namespace, vcat(wtas, n_ff_inh), sys, bc, kwargs)
    end
end

struct LIFExciCircuitBlox <: CompositeBlox
    namespace
    parts
    system
    connector
    kwargs

    function LIFExciCircuitBlox(;
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

struct LIFInhCircuitBlox <: CompositeBlox
    namespace
    parts
    system
    connector
    kwargs

    function LIFInhCircuitBlox(;
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
