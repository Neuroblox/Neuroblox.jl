struct CorticalBlox{P} <: AbstractComponent
    namespace
    parts::Vector{P}
    odesystem
    connector
    mean::Vector{Num}

    function CorticalBlox(;
        name, 
        N_wta,
        namespace=nothing,
        density_wta=0.1,
        N_exci=5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=3.0,
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=5,
        τ_inhib=70
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
                I_in = rand(N_exci),
                freq,
                phase,
                τ_exci,
                τ_inhib    
            )
        end

        g = MetaDiGraph()
        add_blox!.(Ref(g), wtas)

        idxs = Base.OneTo(N_wta)
        for i in idxs
            add_edge!.(Ref(g), i, setdiff(idxs, i), Ref(Dict(:weight => 1.0, :density => density_wta)))
        end

        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(wtas; name)

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

        new{eltype(wtas)}(namespace, wtas, sys, bc, m)
    end
end
