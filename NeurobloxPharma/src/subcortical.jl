"""
    Subcortical blox
    all subcortical blox used in cortico-striatal model are defined here

"""

# internal connectivity matrix
function subcortical_connection_matrix(density, N, weight; rng=Random.default_rng())
    connection_matrix = zeros(N, N)
    idxs = 1:N
    for i in idxs
        for j in idxs
            if !(i==j) && (rand(rng)<=density && connection_matrix[j, i] == 0)
                connection_matrix[i, j] = weight 
            end
        end
    end
    connection_matrix
end

struct Striatum <: AbstractComposite
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
        τ_inhib=70
    )
        n_inh = [
            HHNeuronInhib(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    G_syn = G_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
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

struct GPi <: AbstractComposite
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
        τ_inhib=70
    )
        n_inh = [
            HHNeuronInhib(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    G_syn = G_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
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


struct GPe <: AbstractComposite
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
        τ_inhib=70
    )
        n_inh = [
            HHNeuronInhib(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    G_syn = G_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
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


struct Thalamus <: AbstractComposite
    namespace
    parts
    system
    connector
    mean
    connection_matrix

    function Thalamus(;
        name, 
        namespace = nothing,
        N_exci = 25,
        E_syn_exci=0,
        G_syn_exci=3,
        I_bg=3*ones(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=5,
        density=0.0,
        weight=1,
        connection_matrix=nothing,
        rng=Random.default_rng()              
    )
        I_bg = I_bg isa Array ? I_bg : fill(I_bg, N_exci)
        n_exci = [
            HHNeuronExci(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    G_syn = G_syn_exci, 
                    τ = τ_exci,
                    I_bg = I_bg[i],
            ) 
            for i in Base.OneTo(N_exci)
        ]
        
        g = MetaDiGraph()
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_exci[i])
        end

        if isnothing(connection_matrix)
            connection_matrix = subcortical_connection_matrix(density, N_exci, weight; rng)
        end
        for i in 1:N_exci
            for j in 1:N_exci
                cij = connection_matrix[i,j]
                if !iszero(cij)
                    add_edge!(g, i, j, Dict(:weight => cij))
                end
            end
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

        new(namespace, parts, sys, bc, m, connection_matrix)
    end
end   

struct STN <: AbstractComposite
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
        τ_exci=5
    )
        n_exci = [
            HHNeuronExci(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    G_syn = G_syn_exci, 
                    τ = τ_exci,
                    I_bg = I_bg[i],
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
