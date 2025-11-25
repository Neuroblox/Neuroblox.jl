function GraphDynamics.system_wiring_rule!(g, 
    HH_src::Union{HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam}, 
    HH_dst::Union{HHNeuronInhib_MSN_Adam, HHNeuronInhib_FSI_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam};
    weight, learning_rule=NoLearningRule(), kwargs...)
    
    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(HH_src.name, "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(HH_dst.name, "spikes_cumulative"))
    
    conn = BasicConnection(weight)

    add_connection!(g, HH_src, HH_dst; conn, weight, learning_rule, kwargs...)
end

function (c::BasicConnection)(HH_src::Union{Subsystem{HHNeuronInhib_FSI_Adam},
                                            Subsystem{HHNeuronInhib_MSN_Adam},
                                            Subsystem{HHNeuronExci_STN_Adam},
                                            Subsystem{HHNeuronInhib_GPe_Adam}}, 
                              HH_dst::Union{Subsystem{HHNeuronInhib_FSI_Adam},
                                            Subsystem{HHNeuronInhib_MSN_Adam},
                                            Subsystem{HHNeuronExci_STN_Adam},
                                            Subsystem{HHNeuronInhib_GPe_Adam}},
                              t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.G * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function (c::BasicConnection)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam},
                              HH_dst::Subsystem{HHNeuronInhib_FSI_Adam}, t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.Gₛ * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function GraphDynamics.system_wiring_rule!(g,
                                           HH_src::HHNeuronInhib_FSI_Adam, 
                                           HH_dst::HHNeuronInhib_FSI_Adam; weight, gap=false, kwargs...)

    if gap
        gap_weight = get(kwargs, :gap_weight, 0.0)
        # Add a forwards GAP connection from src to dst
        add_connection!(g, HH_src, HH_dst; conn=HHConnection_GAP(gap_weight))
        # Add a reverse GAP connection from the dst to the src so that its I_gap is modified too
        add_connection!(g, HH_dst, HH_src; conn=HHConnection_GAP_Reverse(gap_weight))
    end
    conn = BasicConnection(weight)
    add_connection!(g, HH_src, HH_dst; conn, weight, gap, kwargs...)
end

function ((;w_gap)::HHConnection_GAP)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam}, 
                                      HH_dst::Subsystem{HHNeuronInhib_FSI_Adam}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap * (HH_dst.V - HH_src.V)
    acc
end

function ((;w_gap_rev)::HHConnection_GAP_Reverse)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam}, 
                                                  HH_dst::Subsystem{HHNeuronInhib_FSI_Adam}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap_rev * (HH_dst.V - HH_src.V)
    acc
end

#----------------------------------------------
# Striatum_MSN_Adam
function GraphDynamics.system_wiring_rule!(g, s::Striatum_MSN_Adam; kwargs...)
    n_inh = s.parts
    connection_matrix = s.connection_matrix
    for n ∈ n_inh
        system_wiring_rule!(g, n)
    end
    for j ∈ axes(connection_matrix, 1)
        for i ∈ axes(connection_matrix, 2)
            cji = connection_matrix[j,i]
            if !iszero(cji)
                system_wiring_rule!(g, n_inh[j], n_inh[i]; weight = cji)
            end
        end
    end
end

#----------------------------------------------
# Striatum_FSI_Adam
function GraphDynamics.system_wiring_rule!(g, s::Striatum_FSI_Adam; kwargs...)
    n_inh = s.parts
    connection_matrix = s.connection_matrix
    for n ∈ n_inh
        system_wiring_rule!(g, n)
    end
    for i ∈ axes(connection_matrix, 2)
        for j ∈ axes(connection_matrix, 1)
            cji = connection_matrix[j, i]
            if iszero(cji.weight) && iszero(cji.g_weight) 
                nothing
            elseif iszero(cji.g_weight) 
                system_wiring_rule!(g, n_inh[j], n_inh[i]; weight=cji.weight)
            else
                system_wiring_rule!(g, n_inh[j], n_inh[i]; weight=cji.weight, gap = true, gap_weight = cji.g_weight)
            end
        end
    end
end


#----------------------------------------------
# GPe_Adam
function GraphDynamics.system_wiring_rule!(g, gpe::GPe_Adam; kwargs...)
    n_inh = gpe.parts
    for n ∈ n_inh
        system_wiring_rule!(g, n)
    end
    connection_matrix = gpe.connection_matrix
    for i ∈ axes(connection_matrix, 1)
        for j ∈ axes(connection_matrix, 2)
            cij = connection_matrix[i,j]
            if !iszero(cij)
                system_wiring_rule!(g, n_inh[i], n_inh[j]; weight = cij)
            end
        end
    end
end

#----------------------------------------------
# STN_Adam
issupported(::STN_Adam) = true
components(stn::STN_Adam) = stn.parts
function GraphDynamics.system_wiring_rule!(g, stn::STN_Adam; kwargs...)
    n_inh = stn.parts
    for n ∈ n_inh
        system_wiring_rule!(g, n)
    end
    connection_matrix = stn.connection_matrix
    for j ∈ axes(connection_matrix, 1)
        for i ∈ axes(connection_matrix, 2)
            cji = connection_matrix[j,i]
            if !iszero(cji)
                system_wiring_rule!(g, n_inh[j], n_inh[i]; weight = cji)
            end
        end
    end
end

#----------------------------------------------
# Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam - Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam

function indegree_constrained_connections!(g,
                                           neurons_src, neurons_dst,
                                           name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = NeurobloxBase.get_density(kwargs, name_src, name_dst)
        NeurobloxBase.indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                system_wiring_rule!(g, neurons_src[i], neurons_dst[j]; kwargs...)
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g,
                                           cb_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                                           cb_dst::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam}; kwargs...)
    neurons_src = cb_src.parts 
    neurons_dst = cb_dst.parts
    indegree_constrained_connections!(g, neurons_src, neurons_dst,
                                      namespaced_nameof(cb_src), namespaced_nameof(cb_dst); kwargs...)
end
