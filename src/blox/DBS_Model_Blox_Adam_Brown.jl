"""
Subcortical Blox used for DBS model in Adam et al,2021 
"""

struct Striatum_MSN_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function Striatum_MSN_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 100,
        E_syn_inhib=-80,
        I_bg=1.172*ones(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=13,
        σ=0.11,
        density=0.3,
        weight=0.1,
        G_M=1.3
    )
        n_inh = [
            HHNeuronInhib_MSN_Adam_Blox(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ,
                    G_M=G_M
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_inhib)
            add_blox!(g, n_inh[i])
        end
        in_degree = Int(ceil(density*(N_inhib)))
        idxs = Base.OneTo(N_inhib)
        for i in idxs
            source_set = setdiff(idxs, i)
            source = sample(source_set, in_degree; replace=false)
            for j in source
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
        end

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
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

struct Striatum_FSI_Adam  <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function Striatum_FSI_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 50,
        E_syn_inhib=-80,
        I_bg=6.2*ones(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=11,
        τ_inhib_s=6.5,
        σ=1.2,
        density=0.58,
        g_density=0.33,
        weight=0.6,
        g_weight=0.15
    )
        n_inh = [
            HHNeuronInhib_FSI_Adam_Blox(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    τ = τ_inhib,
                    τₛ = τ_inhib_s,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_inhib)
            add_blox!(g, n_inh[i])
        end
        in_degree = Int(ceil(density*(N_inhib)))
        gap_degree = Int(ceil(g_density*(N_inhib)))
        idxs = Base.OneTo(N_inhib)

        gap_junctions = zeros(Int, N_inhib)
        for i in idxs
            if gap_junctions[i]<gap_degree
                other_fsi = setdiff(idxs,i)
                rem = findall(x -> x < gap_degree, gap_junctions[other_fsi])
                gap_idx = sample(rem, min(gap_degree, length(rem)); replace=false)
                gap_nbr = other_fsi[gap_idx]
                gap_junctions[i] += length(gap_idx)
                gap_junctions[gap_nbr] .+= 1

            else
                gap_nbr = []
            end
            source_set = setdiff(idxs, i)
            syn_source = sample(source_set, in_degree; replace=false)
            only_syn=setdiff(syn_source,gap_nbr)
            only_gap=setdiff(gap_nbr,syn_source)
            syn_gap=intersect(syn_source,gap_nbr)
            for j in only_syn
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
            for j in only_gap
                add_edge!(g, j, i, Dict(:weight=>0, :gap => true, :gap_weight => g_weight/gap_degree))
            end

            for j in syn_gap
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree, :gap => true, :gap_weight => g_weight/gap_degree))
            end

        end

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
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

struct GPe_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function GPe_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 80,
        E_syn_inhib=-80,
        I_bg=3.4*ones(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=10,
        σ=1.7,
        density=0.0,
        weight=0.0,
    )
        n_inh = [
            HHNeuronInhib_MSN_Adam_Blox(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_inhib)
            add_blox!(g, n_inh[i])
        end
        in_degree = Int(ceil(density*(N_inhib)))
        idxs = Base.OneTo(N_inhib)
        for i in idxs
            source_set = setdiff(idxs, i)
            source = sample(source_set, in_degree; replace=false)
            for j in source
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
        end

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
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

struct STN_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function STN_Adam(;
        name, 
        namespace = nothing,
        N_exci = 40,
        E_syn_exci=0.0,
        I_bg=1.8*ones(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=2,
        σ=1.7,
        density=0.0,
        weight=0.0
    )
        n_exci = [
            HHNeuronExci_STN_Adam_Blox(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    τ = τ_exci,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ
            ) 
            for i in Base.OneTo(N_exci)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_exci[i])
        end
        in_degree = Int(ceil(density*(N_exci)))
        idxs = Base.OneTo(N_exci)
        for i in idxs
            source_set = setdiff(idxs, i)
            source = sample(source_set, in_degree; replace=false)
            for j in source
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
        end

        parts = n_exci

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
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