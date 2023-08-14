# creates a winner-take-all local circuit found in neocortex
# typically 5 pyramidal (excitatory) neurons send synapses to a single interneuron (inhibitory)
# and recieve feedback inhibition from that interneuron
@parameters t
D = Differential(t)

mutable struct WinnerTakeAllBlox{N, P, S, C} <: AbstractComponent
    namespace::N
    parts::Vector{P}
    odesystem::S
    connector::C
    P_connect::Float64

    function WinnerTakeAllBlox(; 
        name, 
        namespace = nothing,
        N_exci = 5,
        P_connect = 0.5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=3.0,
        I_in=zeros(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
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
                    I_in = I_in[i],
                    freq = freq[i],
                    phase = phase[i]
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

        new{
            typeof(namespace), 
            Union{eltype(n_excis), typeof(n_inh)}, 
            typeof(sys), 
            typeof(bc)}(namespace, parts, sys, bc, P_connect)
    end 

end