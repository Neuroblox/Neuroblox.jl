struct CorticalBlox <: CompositeBlox
    namespace
    parts
    odesystem
    connector

    function CorticalBlox(;
        name, 
        N_wta,
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
        wtas = map(Base.OneTo(N_wta)) do i
            WinnerTakeAllBlox(;
                name=Symbol("wta$i"), 
                namespace=namespaced_name(namespace, name),
                N_exci,
                E_syn_exci,
                E_syn_inhib,
                G_syn_exci,
                G_syn_inhib,
                I_bg = I_bg_ar,
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

        idxs = Base.OneTo(N_wta)
        for i in idxs
            add_edge!.(Ref(g), i, setdiff(idxs, i), Ref(Dict(kwargs)))
            add_edge!(g, N_wta+1, i, Dict(:weight => 1))
        end

        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(vcat(wtas, n_ff_inh); name)

        new(namespace, vcat(wtas, n_ff_inh), sys, bc)
    end
end
