"""
    Subcortical blox


"""
struct Striatum{N, P, S, C} <: AbstractComponent
    namespace::N
    parts::Vector{P}
    odesystem::S
    connector::C
    mean::Vector{Num}
    out_degree::Int

    function Striatum(;
        name, 
        namespace = nothing,
        N_inhib = 25,
        out_degree=1,
        E_syn_inhib=-70,
        G_syn_inhib=1.2,
        I_in=zeros(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=70
    )
    n_inh = [
        HHNeuronInhibBlox(
                name = Symbol("inh$i"),
                namespace = namespaced_name(namespace, name), 
                E_syn = E_syn_inhib, 
                G_syn = G_syn_inhib, 
                τ = τ_inhib,
                I_in = I_in[i],
                freq = freq[i],
                phase = phase[i]
        ) 
        for i in Base.OneTo(N_inhib)
    ]

    g = MetaDiGraph()
    for i in Base.OneTo(N_inhib)
        add_blox!(g, n_inh[i])
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
    
    # TO DO : m is a subset of states to be plotted in the GUI. 
    # This can be moved to NeurobloxGUI, maybe via plotting recipes, 
    # since it is not an essential part of the blox.
    m = if isnothing(namespace) 
        [s for s in states.((sys,), states(sys)) if contains(string(s), "V(t)")]
    else
        @variables t
        # HACK : Need to define an empty system to add the correct namespace to states.
        # Adding a dispatch `ModelingToolkit.states(::Symbol, ::AbstractArray)` upstream will solve this.
        sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
        [s for s in states.((sys_namespace,), states(sys)) if contains(string(s), "V(t)")]
    end

    new{typeof(namespace), eltype(wtas), typeof(sys), typeof(bc)}(namespace, parts, sys, bc, m, out_degree)

    end

end    
