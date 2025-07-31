"""
    Subcortical blox
    all subcprtical blox used in cortico-striatal model are defined here

"""
struct Striatum <: CompositeBlox
    namespace
    parts
    system
    connector
    mean

    function Striatum(;
        name, 
        namespace = nothing,
        N_inhib = 25,
        E_syn_inhib=-70,
        G_syn_inhib=1.2,
        I_bg=zeros(N_inhib),
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
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i]
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        matrisome = Matrisome(; name=:matrisome, namespace=namespaced_name(namespace, name))
        striosome = Striosome(; name=:striosome, namespace=namespaced_name(namespace, name))
        
        parts = vcat(n_inh, matrisome, striosome) 

        g = MetaDiGraph()
        add_blox!.(Ref(g), n_inh)

        # If this blox is simulated on its own, 
        # then only the parts with dynamics are included in the system.
        # This is done to avoid messing with structural_simplify downstream. 
        # Also it makes sense, as the discrete parts rely exclusively on inputs/outputs, 
        # which are not present in this case.
        if !isnothing(namespace)
            add_blox!(g, matrisome)
            add_blox!(g, striosome)
            bc = connectors_from_graph(g)
            sys = system_from_parts(parts; name)

            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            m = [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
            
            new(namespace, parts, sys, bc, m)
        else
            bc = connectors_from_graph(g)
            sys = system_from_graph(g, bc; name, simplify=false)

            m = [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
            
            new(namespace, parts, sys, bc, m)
        end
    end
end    

function get_striosome(str::Striatum)
    idx = findfirst(x -> x isa Striosome, str.parts)
    return str.parts[idx]
end

function get_matrisome(str::Striatum)
    idx = findfirst(x -> x isa Matrisome, str.parts)
    return str.parts[idx]
end

struct GPi <: CompositeBlox
    namespace
    parts
    system
    connector
    mean

    function GPi(;
        name, 
        namespace = nothing,
        N_inhib = 25,
        E_syn_inhib=-70,
        G_syn_inhib=8,
        I_bg=4*ones(N_inhib),
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
                    I_bg = I_bg[i],
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
        
        bc = connectors_from_graph(g)
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    


struct GPe <: CompositeBlox
    namespace
    parts
    system
    connector
    mean

    function GPe(;
        name, 
        namespace = nothing,
        N_inhib = 15,
        E_syn_inhib=-70,
        G_syn_inhib=3,
        I_bg=2*ones(N_inhib),
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
                    I_bg = I_bg[i],
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
        
        bc = connectors_from_graph(g)
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    


struct Thalamus <: CompositeBlox
    namespace
    parts
    system
    connector
    mean

    function Thalamus(;
        name, 
        namespace = nothing,
        N_exci = 25,
        E_syn_exci=0,
        G_syn_exci=3,
        I_bg=3*ones(N_exci),
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
                    I_bg = I_bg[i],
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
        
        bc = connectors_from_graph(g)
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end
end   


struct STN <: CompositeBlox
    namespace
    parts
    system
    connector
    mean

    function STN(;
        name, 
        namespace = nothing,
        N_exci = 25,
        E_syn_exci=0,
        G_syn_exci=3,
        I_bg=3*ones(N_exci),
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
                    I_bg = I_bg[i],
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
        
        bc = connectors_from_graph(g)
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)
        
        # TO DO : m is a subset of unknowns to be plotted in the GUI. 
        # This can be moved to NeurobloxGUI, maybe via plotting recipes, 
        # since it is not an essential part of the blox.
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            # HACK : Need to define an empty system to add the correct namespace to unknowns.
            # Adding a dispatch `ModelingToolkit.unknowns(::Symbol, ::AbstractArray)` upstream will solve this.
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end
end    
