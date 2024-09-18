"""
    WinnerTakeAllBlox

Creates a winner-take-all local circuit found in neocortex,
typically 5 pyramidal (excitatory) neurons send synapses to a single interneuron (inhibitory)
and receive feedback inhibition from that interneuron.
"""
struct WinnerTakeAllBlox{P} <: AbstractCompositeBlox
    namespace
    parts::Vector{P}
    odesystem
    connector

    function WinnerTakeAllBlox(; 
        name, 
        namespace = nothing,
        N_exci = 5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=4.0,
        I_bg=zeros(N_exci),
        freq=0.0,
        phase=0.0,
        τ_exci=5,
        τ_inhib=70
    )  
        n_inh = HHNeuronInhibBlox(
            name = "inh",
            namespace = namespaced_name(namespace, name), 
            E_syn = E_syn_inhib, 
            G_syn = G_syn_inhib, 
            τ = τ_inhib
        )
        n_excis = [
            HHNeuronExciBlox(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    G_syn = G_syn_exci, 
                    τ = τ_exci,
                    I_bg = (I_bg isa Array) ? I_bg[i] : I_bg*rand(), # behave differently if I_bg is array
                    freq = freq,
                    phase = phase
            ) 
            for i in Base.OneTo(N_exci)
        ]

        g = MetaDiGraph()
        add_blox!(g, n_inh)
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_excis[i])
            add_edge!(g, 1, i+1, :weight, 1.0)
            add_edge!(g, i+1, 1, :weight, 1.0)
        end

        parts = vcat(n_inh, n_excis)
        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)

        new{Union{eltype(n_excis), typeof(n_inh)}}(namespace, parts, sys, bc)
    end 

end
