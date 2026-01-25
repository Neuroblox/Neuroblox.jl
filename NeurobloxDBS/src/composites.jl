function adam_connection_matrix(density, N, weight)
    connection_matrix = zeros(N, N)
    in_degree = Int(ceil(density*(N)))
    idxs = 1:N
    for i in idxs
        source_set = setdiff(idxs, i)
        source = sample(source_set, in_degree; replace=false)
        for j in source
            connection_matrix[j, i] = weight / in_degree
        end
    end
    connection_matrix
end

function adam_connection_matrix_gap(density, g_density, N, weight, g_weight)
    connection_matrix = [(weight = 0.0, g_weight = 0.0) for _ ∈ 1:N, _ ∈ 1:N]
    in_degree = Int(ceil(density*N))
    gap_degree = Int(ceil(g_density*N))
    idxs = 1:N
    gap_junctions = zeros(Int, N)
    for i in idxs
        if gap_junctions[i] < gap_degree
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
            connection_matrix[j, i] = (;weight = weight/in_degree, g_weight=0)
        end
        for j in only_gap
            connection_matrix[j, i] = (;weight = 0, g_weight=g_weight/gap_degree)
        end
        for j in syn_gap
           connection_matrix[j, i] = (;weight = weight/in_degree, g_weight=g_weight/gap_degree)
        end
    end
    connection_matrix
end

struct Striatum_MSN_Adam <: AbstractComposite
    name
    namespace
    parts
    connection_matrix
    graph::GraphSystem

    function Striatum_MSN_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 100,
        E_syn_inhib=-80,
        I_bg=1.172*ones(N_inhib),
        τ_inhib=13,
        σ=0.11,
        density=0.3,
        weight=0.1,
        G_M=1.3,
        connection_matrix=nothing
    )
        if isnothing(connection_matrix)
            connection_matrix = adam_connection_matrix(density, N_inhib, weight)
        end

        sys = @graph begin
            @nodes begin
                n_inh = [
                    HHNeuronInhib_MSN_Adam(
                            name = Symbol("inh$i"),
                            namespace = namespaced_name(namespace, name),
                            E_syn = E_syn_inhib, 
                            τ = τ_inhib,
                            I_bg = I_bg[i],
                            σ=σ,
                            G_M=G_M
                    ) 
                    for i in 1:N_inhib
                ]
            end

            @connections begin
                for i ∈ axes(connection_matrix, 2)
                    for j ∈ axes(connection_matrix, 1)
                        cji = connection_matrix[j,i]
                        if !iszero(cji)
                            n_inh[j] => n_inh[i], [weight = cji]
                        end
                    end
                end
            end
        end

        new(name, namespace, n_inh, connection_matrix, sys)
    end
end    

struct Striatum_FSI_Adam  <: AbstractComposite
    name
    namespace
    parts
    connection_matrix
    graph::GraphSystem

    function Striatum_FSI_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 50,
        E_syn_inhib=-80,
        I_bg=6.2*ones(N_inhib),
        τ_inhib=11,
        τ_inhib_s=6.5,
        σ=1.2,
        density=0.58,
        g_density=0.33,
        weight=0.6,
        g_weight=0.15,
        connection_matrix=nothing
    )

        if isnothing(connection_matrix)
            connection_matrix = adam_connection_matrix_gap(density, g_density, N_inhib, weight, g_weight)
        end

        sys = @graph begin
            @nodes begin
                n_inh = [
                    HHNeuronInhib_FSI_Adam(
                            name = Symbol("inh$i"),
                            namespace = namespaced_name(namespace, name), 
                            E_syn = E_syn_inhib, 
                            τ = τ_inhib,
                            τₛ = τ_inhib_s,
                            I_bg = I_bg[i],
                            σ=σ
                    ) 
                    for i in 1:N_inhib
                ]
            end

            @connections begin
                for i ∈ axes(connection_matrix, 2)
                    for j ∈ axes(connection_matrix, 1)
                        cji = connection_matrix[j, i]
                        w = cji.weight
                        gw = cji.g_weight

                        if iszero(w) && iszero(gw) 
                        elseif iszero(gw) 
                            n_inh[j] => n_inh[i], [weight = w]
                        else
                            n_inh[j] => n_inh[i], [weight = w, gap = true, gap_weight = gw]
                        end
                    end
                end
            end
        end

        new(name, namespace, n_inh, connection_matrix, sys)
    end
end

struct GPe_Adam <: AbstractComposite
    name
    namespace
    parts
    connection_matrix
    graph::GraphSystem

    function GPe_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 80,
        E_syn_inhib=-80,
        I_bg=3.4*ones(N_inhib),
        τ_inhib=10,
        σ=1.7,
        density=0.0,
        weight=0.0,
        connection_matrix=nothing
    )
        if isnothing(connection_matrix)
            connection_matrix = adam_connection_matrix(density, N_inhib, weight)
        end

        sys = @graph begin
            @nodes begin
                n_inh = [
                    HHNeuronInhib_MSN_Adam(
                            name = Symbol("inh$i"),
                            namespace = namespaced_name(namespace, name), 
                            E_syn = E_syn_inhib, 
                            τ = τ_inhib,
                            I_bg = I_bg[i],
                            σ=σ
                    ) 
                    for i in 1:N_inhib
                ]
            end
            @connections begin
                for i ∈ axes(connection_matrix, 2)
                    for j ∈ axes(connection_matrix, 1)
                        cji = connection_matrix[j,i]

                        if !iszero(cji)
                            n_inh[j] => n_inh[i], [weight = cji]
                        end
                    end
                end
            end
        end

        new(name, namespace, n_inh, connection_matrix, sys) 
    end
end    

struct STN_Adam <: AbstractComposite
    name
    namespace
    parts
    connection_matrix
    graph::GraphSystem

    function STN_Adam(;
        name, 
        namespace = nothing,
        N_exci = 40,
        E_syn_exci=0.0,
        I_bg=1.8*ones(N_exci),
        τ_exci=2,
        σ=1.7,
        density=0.0,
        weight=0.0,
        connection_matrix=nothing
    )

        if isnothing(connection_matrix)
            connection_matrix = adam_connection_matrix(density, N_exci, weight)
        end
        sys = @graph begin
            @nodes begin
                n_exci = [
                    HHNeuronExci_STN_Adam(
                            name = Symbol("exci$i"),
                            namespace = namespaced_name(namespace, name), 
                            E_syn = E_syn_exci, 
                            τ = τ_exci,
                            I_bg = I_bg[i],
                            σ=σ
                    ) 
                    for i in 1:N_exci
                ]
            end
            @connections begin
                for i ∈ axes(connection_matrix, 2)
                    for j ∈ axes(connection_matrix, 1)
                        cji = connection_matrix[j,i]

                        if !iszero(cji)
                            n_exci[j] => n_exci[i], [weight = cji]
                        end
                    end
                end
            end
        end

        new(name, namespace, n_exci, connection_matrix, sys)
    end
end
