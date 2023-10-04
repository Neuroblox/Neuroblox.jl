"""
    Subcortical blox
    all subcprtical blox used in cortico-striatal model are defined here

"""
struct Striatum <: AbstractComponent
    namespace
    parts
    odesystem
    connector
    mean

    function Striatum(;
        name, 
        namespace = nothing,
        N_inhib = 25,
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

    matrisome = DiscreteSpikes(; name=:matrisome, namespace=namespaced_name(namespace, name))
    striosome = DiscreteSpikes(; name=:striosome, namespace=namespaced_name(namespace, name))

    g = MetaDiGraph()
    add_blox!.(Ref(g), n_inh)
    add_blox!(g, matrisome)
    add_blox!(g, striosome)

    parts = [n_inh, matrisome, striosome]
    bc = BloxConnector()
  
    sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
    
    m = if isnothing(namespace) 
        [s for s in states.((sys,), states(sys)) if contains(string(s), "V(t)")]
    else
        @variables t
        sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
        [s for s in states.((sys_namespace,), states(sys)) if contains(string(s), "V(t)")]
    end

    new(namespace, parts, sys, bc, m)

    end
end    



struct GPi <: AbstractComponent
    namespace
    parts
    odesystem
    connector
    mean

    function GPi(;
        name, 
        namespace = nothing,
        N_inhib = 25,
        out_degree=1,
        E_syn_inhib=-70,
        G_syn_inhib=8,
        I_in=4*ones(N_inhib),
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

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in states.((sys,), states(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in states.((sys_namespace,), states(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    


struct GPe <: AbstractComponent
    namespace
    parts
    odesystem
    connector
    mean

    function GPe(;
        name, 
        namespace = nothing,
        N_inhib = 15,
        out_degree=1,
        E_syn_inhib=-70,
        G_syn_inhib=3,
        I_in=2*ones(N_inhib),
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

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in states.((sys,), states(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in states.((sys_namespace,), states(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)

    end

end    


struct Thalamus <: AbstractComponent
    namespace
    parts
    odesystem
    connector
    mean

    function Thalamus(;
        name, 
        namespace = nothing,
        N_exci = 25,
        out_degree=1,
        E_syn_exci=0,
        G_syn_exci=3,
        I_in=3*ones(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=5
    )
        n_exci = [
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
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_exci[i])
        end

        parts = n_exci
    
        bc = connector_from_graph(g)
        
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in states.((sys,), states(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in states.((sys_namespace,), states(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end   


struct STN <: AbstractComponent
    namespace
    parts
    odesystem
    connector
    mean

    function STN(;
        name, 
        namespace = nothing,
        N_exci = 25,
        out_degree=1,
        E_syn_exci=0,
        G_syn_exci=3,
        I_in=3*ones(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=5
    )
    n_exci = [
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
    for i in Base.OneTo(N_exci)
        add_blox!(g, n_exci[i])
    end

    parts = n_exci
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

    new(namespace, parts, sys, bc, m)

    end

end    