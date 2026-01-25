function GraphDynamics.system_wiring_rule!(g, blox_src::Union{HHNeuronExci, HHNeuronInhib}, blox_dst::Union{HHNeuronExci, HHNeuronInhib}; kwargs...)
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse(blox_src, blox_dst; kwargs...)
    synapse_wiring_rule!(g, blox_src, syn, blox_dst; kwargs...)
end

function synapse_wiring_rule!(g, blox_src::HHNeuronExci, syn_ampa::Glu_AMPA_Synapse, blox_dst::Union{HHNeuronInhib, HHNeuronExci};
                              learning_rule=NoLearningRule(), sta = false, synapse_kwargs=(;), kwargs...)
    @graph! g begin
        weight = get_weight(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst))
        learning_rule = maybe_set_state_pre(learning_rule, blox_src.spikes_cumulative)
        if blox_dst isa HHNeuronExci
            learning_rule = maybe_set_state_post(learning_rule, blox_dst.spikes_cumulative)
        end
        if sta
            #============================
            STA synapses require both the pre-synaptic voltage to calculate z, and the post synaptic voltage
            to calculate zₛₜₚ. Information from both are required to create the I_syn that gets sent to the
            post-synaptic neuron.
            Hence, we make
            * one Glu_AMPA_Synapse that originates from the presynaptic neuron that calculates G and z based on the presynaptic voltage
            * a Glu_AMPA_STA_Synapse originating from the postsynaptic neuron that calculates Gₛₜₚ and zₛₜₚ based on the postsynaptic voltage

            We then send a MultipointConnection from the presynaptic  neuron to the postsynaptic neuron that
            has access to both synapses so they can be used to compute I_syn

            # [::HHNeuronExci] ------------------------I_syn--------------------------> [::HHneuronExci]
            #                \                         ↑   ↑                               /
            #                 \--V-->[::Glu_AMPA_Synapse]  [::Glu_AMPA_STA_Synapse] <--V--/

            This design lets both the Glu_AMPA_Synapse and the Glu_AMPA_STA_Synapses be re-usable!
            ============================#
            syn_sta = Glu_AMPA_STA_Synapse(blox_src, blox_dst; synapse_kwargs...)
            @connections begin
                @rule blox_src => syn_ampa, [ignore_if_exists=true, kwargs..., weight=1.0]
                @rule blox_dst => syn_sta,  [ignore_if_exists=true, kwargs..., weight=1.0]

                conn=MultipointConnection(weight, (syn_ampa=PartitionedIndex(g, syn_ampa),
                                                   syn_sta =PartitionedIndex(g, syn_sta)))
                
                blox_src => blox_dst, [kwargs..., learning_rule, weight, conn]
            end
        else
            @connections begin
                # If we're re-using a synapse, don't re-add the connection
                @rule blox_src => syn_ampa, [ignore_if_exists=true, kwargs...]
                conn=MultipointConnection(weight, (;syn=PartitionedIndex(g, syn_ampa)))
                blox_src => blox_dst, [kwargs..., conn, learning_rule]
            end
        end
    end
end

function synapse_wiring_rule!(g, blox_src::HHNeuronInhib, syn::GABA_A_Synapse, blox_dst::Union{HHNeuronExci, HHNeuronInhib}; kwargs...)
    weight = get_weight(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst))
    @graph! g begin
        @connections begin
            # If we're re-using a synapse, don't re-add the connection
            @rule blox_src => syn, [ignore_if_exists=true, kwargs..., weight=1]
            conn=MultipointConnection(weight, (;syn=PartitionedIndex(g, syn)))
            blox_src => blox_dst, [kwargs..., conn]
        end
    end
end

function (::BasicConnection)(sys_src::Subsystem{HHNeuronExci}, sys_dst::Subsystem{<:Union{Glu_AMPA_Synapse, Glu_AMPA_STA_Synapse}}, t)
    input = initialize_input(sys_dst)
    @reset input.V = sys_src.V
end

function (::BasicConnection)(sys_src::Subsystem{HHNeuronInhib}, sys_dst::Subsystem{<:Union{GABA_A_Synapse}}, t)
    input = initialize_input(sys_dst)
    @reset input.V = sys_src.V
end

function (c::MultipointConnection)(sys_src::Subsystem,
                                   sys_syn::Subsystem{<:Union{GABA_A_Synapse, Glu_AMPA_Synapse}},
                                   sys_dst::Subsystem{<:Union{HHNeuronExci, HHNeuronInhib}},
                                   t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_syn.g * sys_syn.G * (sys_dst.V - sys_syn.E_syn)
end

function (conn::MultipointConnection)(sys_src ::Subsystem{HHNeuronExci},
                                      syn_ampa::Subsystem{Glu_AMPA_Synapse},
                                      syn_sta ::Subsystem{Glu_AMPA_STA_Synapse},
                                      sys_dst ::Subsystem{HHNeuronExci},
                                      t)
    input = initialize_input(sys_dst)
    w = conn.weight
    @reset input.I_syn = -w * syn_sta.g * syn_ampa.G * syn_sta.Gₛₜₚ * (sys_dst.V - syn_ampa.E_syn)
end

function (c::BasicConnection)(sys_src::Subsystem{NGNMM_theta}, sys_dst::Subsystem{<:Union{HHNeuronExci, HHNeuronInhib}}, t)
    w = c.weight
    a = sys_src.aₑ
    b = sys_src.bₑ
    acc = initialize_input(sys_dst)
    @reset acc.I_asc = w * (1/(sys_src.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)
end

##----------------------------------------------

# FSI stuff, TODO: receptor blox for FSI neurons?
@wiring_rule (blox_src::Union{HHNeuronExci, HHNeuronInhib}, blox_dest::HHNeuronFSI; sta=false, gap=false, kwargs...) begin
    if sta
        error("STA connections with HHNeuronFSI are not implemented")
    end
    if gap
        error("GAP connections with HHNeuronFSI are not implemented")
    end
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse(blox_src; kwargs...)
    # wire up the src to syn and syn to dst
    @connections begin
        # If we're re-using a synapse, don't re-add the connection
        @rule blox_src => syn, [ignore_if_exists=true, kwargs..., weight=1]

        conn=MultipointConnection(weight, (;syn=PartitionedIndex(__graph__, syn)))
        blox_src => blox_dst, [kwargs..., conn]
    end
end
@wiring_rule (blox_src::HHNeuronFSI, blox_dest::HHNeuronFSI; sta=false, gap=false, kwargs...) begin
    if sta
        error("STA connections with HHNeuronFSI are not implemented")
    end
    if gap
        error("GAP connections with HHNeuronFSI are not implemented")
    end
    weight = get_weight(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst))
    conn = BasicConnection(weight)
    @connections begin
        blox_src => blox_dst, [kwargs..., conn]
    end
end

function (conn::BasicConnection)(sys_src::Subsystem,
                                 sys_syn::Subsystem{<:Union{GABA_A_Synapse, Glu_AMPA_Synapse}},
                                 sys_dst::Subsystem{HHNeuronFSI},
                                 t)
    acc = initialize_input(sys_dst)
    w = conn.weight
    @set acc.I_syn = -w * sys_syn.G * (sys_syn.V - sys_syn.E_syn)
end
function (conn::BasicConnection)(sys_src::Subsystem{HHNeuronFSI},
                                 sys_dst::Subsystem{<:Union{HHNeuronExci, HHNeuronInhib}},
                                 t)
    acc = initialize_input(sys_dst)
    w = conn.weight
    @set acc.I_syn = -w * sys_src.G * (sys_dst.V - sys_src.E_syn)
end
function (conn::BasicConnection)(sys_src::Subsystem{HHNeuronFSI},
                                 sys_dst::Subsystem{HHNeuronFSI},
                                 t)
    acc = initialize_input(sys_dst)
    w = conn.weight
    @set acc.I_syn = -w * sys_src.Gₛ * (sys_dst.V - sys_src.E_syn)
end

##----------------------------------------------
# WinnerTakeAll

@wiring_rule (blox_src::WinnerTakeAll, blox_dst::WinnerTakeAll; kwargs...) begin
    # users can supply a :connection_matrix to the graph edge, where
    # connection_matrix[i, j] determines if neurons_src[i] is connected to neurons_src[j]
    connection_matrix = get_connection_matrix(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst),
                                              length(blox_src.excis), length(blox_dst.excis))
    @connections begin
        for (j, neuron_postsyn) in enumerate(blox_dst.excis)
            name_postsyn = namespaced_nameof(neuron_postsyn)
            for (i, neuron_presyn) in enumerate(blox_src.excis)
                name_presyn = namespaced_nameof(neuron_presyn)
                if (name_postsyn != name_presyn) && connection_matrix[i, j]
                    @rule neuron_presyn => neuron_postsyn, [kwargs...]
                end
            end
        end
    end
end

@wiring_rule (blox_src::HHNeuronInhib, blox_dst::WinnerTakeAll; kwargs...) begin
    @connections begin
        for neuron_postsyn in blox_dst.excis
            @rule blox_src => neuron_postsyn, [kwargs...]
        end
    end
end

#---------------------------------------------------------------------
# Cortical Blox, STN, Thalamus

function sparse_projection_connections!(g, cort_src::Cortical, cort_dst::Cortical;
                                        indegree, weight, density, rng=default_rng(), kwargs...)
    excis_src = get_exci_neurons(cort_src)
    excis_dst = get_exci_neurons(cort_dst)
    N_src = length(excis_src)
    N_dst = length(excis_dst)

    CC_mat = zeros(N_src, N_dst)
    for j ∈ 1:N_dst
        if rand(rng) < density
            is = randperm(rng, N_src)
            CC_mat[is[1:indegree], j] .= weight
        end
    end
    weight_matrix_connections!(g, excis_src, excis_dst, namespaced_nameof(cort_src), namespaced_nameof(cort_dst); kwargs..., weightmatrix=CC_mat)
end

function sparse_projection_connections!(g, thal_src::Thalamus, cort_dst::Cortical;
                                        modulator::Cortical, boosted_weight, inhibited_weight, density, rng=default_rng(), kwargs...)
    excis_mod = get_exci_neurons(modulator)
    excis_src = get_exci_neurons(thal_src)
    excis_dst = get_exci_neurons(cort_dst)
    N_src = length(excis_src)
    N_dst = length(excis_dst)

    TC_mat = zeros(N_src, N_dst)
    for j ∈ 1:N_dst
        if rand(rng) < density
            weight = if any(exci_src -> has_connection(g, exci_src, excis_dst[j]), excis_mod)
                boosted_weight
            else
                inhibited_weight
            end
            TC_mat[rand(rng, 1:N_src), j] = weight
        end
    end
    weight_matrix_connections!(g, excis_src, excis_dst, namespaced_nameof(thal_src), namespaced_nameof(cort_dst); kwargs..., weightmatrix=TC_mat)
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Cortical,
                                           blox_dst::Cortical;
                                           kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    if get(kwargs, :connection_rule, nothing) == :sparse_projection
        sparse_projection_connections!(g, blox_src, blox_dst; kwargs...)
    elseif haskey(kwargs, :weightmatrix)
        weight_matrix_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                   kwargs...)
    else
        hypergeometric_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                    kwargs...)
    end
end
function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Thalamus,
                                           blox_dst::Cortical;
                                           kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    if get(kwargs, :connection_rule, nothing) == :modulated_sparse_projection
        sparse_projection_connections!(g, blox_src, blox_dst; kwargs...)
    elseif haskey(kwargs, :weightmatrix)
        weight_matrix_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                   kwargs...)
    else
        hypergeometric_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                    kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{Cortical,STN,Thalamus},
                                           blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    if haskey(kwargs, :weightmatrix)
        weight_matrix_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                   kwargs...)
    else
        hypergeometric_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst);
                                    kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_inh_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::NGNMM_theta, blox_dst::Cortical; kwargs...)
    neurons_dst = get_inh_neurons(blox_dst)
    system_wiring_rule!(g, blox_src, blox_dst.n_ff_inh; kwargs...)
end


function (c::BasicConnection)(src::Subsystem{<:AbstractSimpleStimulus}, dst::Subsystem{<:Union{HHNeuronInhib, HHNeuronExci}}, t)
    w = c.weight
    input = initialize_input(dst)
    x = only(outputs(src))
    @reset input.I_in = w * x
end

function GraphDynamics.system_wiring_rule!(g, src::AbstractStimulus, dst::Thalamus; kwargs...) 
    for neuron_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, neuron_dst; kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g, cb::Cortical, str::Striatum; kwargs...)
    neurons_dst = get_inh_neurons(str)
    neurons_src = get_exci_neurons(cb)
    
    w = get_weight(kwargs, cb.name, str.name)

    dist = Uniform(0, 1)
    wt_ar = 2*w*rand(dist, length(neurons_src)) # generate a uniform distribution of weight with average value w
    kwargs = (kwargs..., weight=wt_ar)
    if haskey(kwargs, :learning_rule)
        lr = kwargs.learning_rule
        matr = str.matrisome
        lr = maybe_set_state_post(lr, matr.H_learning)
        kwargs = (kwargs..., learning_rule=lr)
    end
    hypergeometric_connections!(g, neurons_src, neurons_dst, cb.name, str.name; kwargs...)

    algebraic_parts = (str.matrisome, str.striosome)
    for (i, neuron_presyn) ∈ enumerate(neurons_src)
        kwargs = (kwargs...,weight=wt_ar[i])
        for part ∈ algebraic_parts
            system_wiring_rule!(g, neuron_presyn, part; kwargs...)
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, sys_src::Striatum, sys_dst::Striatum; kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    system_wiring_rule!(g, sys_src.matrisome, sys_dst.matrisome; t_event=t_event +   √(eps(t_event)), kwargs...)
    system_wiring_rule!(g, sys_src.matrisome, sys_dst.striosome; t_event=t_event + 2*√(eps(t_event)), kwargs...)
    for inhib ∈ sys_dst.inhibs
        system_wiring_rule!(g, sys_src.matrisome, inhib; t_event=t_event+2*√(eps(t_event)), kwargs...)
    end
    nothing
end

function GraphDynamics.system_wiring_rule!(g, sys_src::TAN, sys_dst::Striatum; kwargs...)
    system_wiring_rule!(g, sys_src, sys_dst.matrisome; kwargs...)
end


function GraphDynamics.system_wiring_rule!(g, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inh_neurons(blox_dst)
    neurons_src = get_inh_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::Union{Cortical,STN,Thalamus}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inh_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::HHNeuronExci, blox_dst::Union{Striatum, GPi}; kwargs...)
    for neuron_dst ∈ get_inh_neurons(blox_dst)
        system_wiring_rule!(g, blox_src, neuron_dst; kwargs...)
    end
end

#----------------------------------------------
# Discrete blox

function GraphDynamics.system_wiring_rule!(g, sys_src::Striatum, sys_dst::Union{TAN, SNc}; kwargs...)
    system_wiring_rule!(g, sys_src.striosome, sys_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, sys_src::Striosome, sys_dst::Union{TAN,SNc};
                           weight, kwargs...)
    conn = BasicConnection(weight)
    add_connection!(g, sys_src, sys_dst; weight, conn, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{Striosome}, sys_dst::Subsystem{<:Union{TAN, SNc}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.H * sys_src.jcn_t_block
end


function GraphDynamics.system_wiring_rule!(g,
                                           sys_src::HHNeuronExci,
                                           sys_dst::Union{Matrisome, Striosome};
                                           weight,
                                           learning_rule=NoLearningRule(),
                                           kwargs...)
    
    conn = BasicConnection(weight)
    learning_rule = maybe_set_state_pre( learning_rule, sys_src.spikes_cumulative)
    learning_rule = maybe_set_state_post(learning_rule, sys_dst.H_learning)
    add_connection!(g, sys_src, sys_dst; weight, conn, learning_rule, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{HHNeuronExci}, sys_dst::Subsystem{<:Union{Matrisome, Striosome}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.spikes_window
end


function GraphDynamics.system_wiring_rule!(g, sys_src::Matrisome, sys_dst::Union{Matrisome, Striosome, HHNeuronInhib}; weight=1.0, kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    conn = EventConnection(weight, (;t_init=0.1, t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end

function (c::EventConnection)(src::Subsystem{Matrisome}, dst::Subsystem{<:Union{Matrisome, Striosome, HHNeuronInhib}}, t)
    initialize_input(dst)
end


function GraphDynamics.apply_discrete_event!(integrator,
                                             sys_view_src,
                                             sys_view_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Matrisome})
    (;params_partitioned, partition_plan, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = partitioned(u, partition_plan)

    (;t_event) = ec.event_times
    if haskey(ec.event_times, :t_init) && t == ec.event_times.t_init
        sys_view_dst.H[] = 1
    end
    if t == t_event
        sys_view_dst.H[] = m_src.ρ > m_dst.ρ ? 0 : 1
    end
    nothing
end

function find_competitor_matrisome(integrator, m::Subsystem{Matrisome}, j)
    (;params_partitioned, partition_plan, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = partitioned(u, partition_plan)
    i = findfirst(v -> eltype(v) <: SubsystemStates{Matrisome}, states_partitioned)
    l = findfirst(eachindex(states_partitioned[i])) do l
        found = false
        if l != j
            for nc ∈ 1:length(connection_matrices)
                M = connection_matrices[nc].data[i][i]
                if !(M isa NotConnected)
                    found = isstored(M, l, j) && isstored(M, j, l)
                end
            end
        end
        found
    end
    if !isnothing(l)
        Subsystem(states_partitioned[i][l], params_partitioned[i][l])
    else
        @warn "No competitor found for" m.name
    end
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             sys_view_src,
                                             sys_view_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{HHNeuronInhib})
    t = integrator.t
    (;t_init, t_event) = ec.event_times
    if t == t_init
        sys_view_dst.I_bg[] = 0.0
    elseif t == t_event
        m_comp = find_competitor_matrisome(integrator, m_src, get_parent_index(sys_view_src))
        if !isnothing(m_comp)
            sys_view_dst.I_bg[] = m_src.ρ > m_comp.ρ ? -2.0 : 0.0
        end
    else
        error("Invalid event time, this shouldn't be possible")
    end
    nothing
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             sys_view_src,
                                             sys_view_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Striosome})
    t = integrator.t
    (;t_init, t_event) = ec.event_times
    if t == t_init
        sys_view_dst.H[] = 1
    else
        m_comp = find_competitor_matrisome(integrator, m_src, get_parent_index(sys_view_src))
        if !isnothing(m_comp)
            sys_view_dst.H[] = m_src.ρ > m_comp.ρ ? 0 : 1
        end
    end
    nothing
end

function GraphDynamics.system_wiring_rule!(g, sys_src::TAN, sys_dst::Matrisome; weight=1.0, kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    conn = EventConnection(weight, (; t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end
function (c::EventConnection)(sys_src::Subsystem{TAN}, sys_dst::Subsystem{Matrisome}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_dst.TAN_spikes
end


function GraphDynamics.apply_discrete_event!(integrator,
                                             sys_view_src,
                                             sys_view_dst,
                                             ec::EventConnection,
                                             sys_src::Subsystem{TAN},
                                             sys_dst::Subsystem{Matrisome})
    w = ec.weight
    sys_view_dst.TAN_spikes[] = w * rand(sys_src.rng, Poisson(sys_src.R))
    nothing
end


#--------------------
# ImageStimulus

function GraphDynamics.system_wiring_rule!(g, stim::ImageStimulus, neuron::Union{HHNeuronInhib, HHNeuronExci}; current_pixel, weight, kwargs...)
    add_connection!(g, stim, neuron; conn=StimConnection(weight, current_pixel), weight, kwargs...)
end

struct StimConnection <: ConnectionRule
    weight::Float64
    pixel_index::Int
end
NeurobloxBase.get_weight((; weight)::StimConnection) = weight
function GraphDynamics.connection_property_namemap(::StimConnection, name_src, name_dst)
    (; weight = Symbol(:w_, name_src, :_, name_dst))
end

function (c::StimConnection)(src::Subsystem{ImageStimulus},
                             dst::Subsystem{<:Union{HHNeuronExci, HHNeuronInhib}},
                             t)
    w = c.weight
    input = initialize_input(dst)
    @reset input.I_in = w * src.current_image[c.pixel_index]
end

function GraphDynamics.system_wiring_rule!(g, src::ImageStimulus, dst::Cortical; kwargs...)
    for n_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, n_dst; current_pixel=src.current_pixel[], kwargs...)
        increment_pixel!(src)
    end
end

function NeurobloxBase.connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, get_matrisome(str1), get_matrisome(str2))
end

function NeurobloxBase.connect_action_selection!(as::AbstractActionSelection, matr1::Matrisome, matr2::Matrisome)
    as.competitor_states = [matr1.ρ_, matr2.ρ_] #HACK : accessing values of rho at a specific time after the simulation
end

# #-------------------------
# NMDA receptor


function synapse_wiring_rule!(g,
    blox_src::Union{HHNeuronExci,HHNeuronInhib},
    blox_syn::Union{MoradiNMDAR,MoradiFullNMDAR},
    blox_dst::Union{HHNeuronExci,HHNeuronInhib}; weight, kwargs...)
    add_node!(g, blox_src)
    add_node!(g, blox_syn)
    add_node!(g, blox_dst)
    @graph! g begin
        @connections begin
            # Inputs to synapse. It needs info from both the presynaptic and postsynaptic neuron
            blox_src => blox_syn, [conn = MultipointConnection(1.0, (; n_dst=PartitionedIndex(g, blox_dst)))]
            # Inputs to postsynaptic neuron.
            blox_src => blox_dst, [kwargs..., conn = MultipointConnection(weight, (; syn=PartitionedIndex(g, blox_syn)))]
        end
    end
end

function (::BasicConnection)(s::Subsystem{VoltageClampSource}, syn::Union{Subsystem{MoradiNMDAR},Subsystem{MoradiFullNMDAR}}, t)
    input = initialize_input(syn)
    @reset input.V_post = s.V
end


# n1---->-------<---n2
#          ↓
#         syn
function (c::MultipointConnection)(sys_src::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    sys_dst::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    sys_syn::Union{Subsystem{MoradiNMDAR},Subsystem{MoradiFullNMDAR}},
    t)
    acc = initialize_input(sys_syn)
    @reset acc.V_pre = sys_src.V
    @reset acc.V_post = sys_dst.V
end

# n1--->---------->n2
#          ↑
#         syn
function (c::MultipointConnection)(sys_src::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    sys_syn::Subsystem{MoradiNMDAR},
    sys_dst::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    t)
    acc = initialize_input(sys_dst)

    Mg = 1 / (1 + sys_syn.Mg_O * exp(-sys_syn.z_Mg * sys_syn.δ * sys_syn.F * sys_dst.V / (sys_syn.R * sys_syn.T)) / sys_syn.IC_50)
    I = -(sys_syn.B - sys_syn.A) * (sys_syn.g_VI + sys_syn.g) * Mg * (sys_dst.V - sys_syn.E)
    @reset acc.I_syn = c.weight * I
end

function (c::MultipointConnection)(sys_src::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    sys_syn::Subsystem{MoradiFullNMDAR},
    sys_dst::Union{Subsystem{HHNeuronExci},Subsystem{HHNeuronInhib}},
    t)
    acc = initialize_input(sys_dst)

    Mg = 1 / (1 + sys_syn.Mg_O * exp(-sys_syn.z_Mg * sys_syn.δ * sys_syn.F * sys_dst.V / (sys_syn.R * sys_syn.T)) / sys_syn.IC_50)
    I = -(sys_syn.w_C * sys_syn.C + sys_syn.w_B * sys_syn.B - sys_syn.A) * sys_syn.g * Mg * (sys_dst.V - sys_syn.E)
    @reset acc.I_syn = c.weight * I
end

function synapse_wiring_rule!(g,
    blox_src::Union{HHNeuronExci,HHNeuronInhib},
    blox_syn::Union{GABA_B_Synapse,NMDA_Synapse},
    blox_dst::Union{HHNeuronExci,HHNeuronInhib}; weight, kwargs...)
    add_node!(g, blox_src)
    add_node!(g, blox_syn)
    add_node!(g, blox_dst)
    @graph! g begin
        @connections begin
            # Inputs to synapse. It needs info from both the presynaptic and postsynaptic neuron
            blox_src => blox_syn, [conn = BasicConnection(1.0)]
            # Inputs to postsynaptic neuron.
            blox_src => blox_dst, [kwargs..., conn = MultipointConnection(weight, (; syn=PartitionedIndex(g, blox_syn)))]
        end
    end
end

function (::BasicConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}}, sys_dst::Subsystem{GABA_B_Synapse}, t)
    input = initialize_input(sys_dst)
    @reset input.V = sys_src.V
end

function (::BasicConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}}, sys_dst::Subsystem{NMDA_Synapse}, t)
    input = initialize_input(sys_dst)
    @reset input.V = sys_src.V
end


# n1--->---------->n2
#          ↑
#         syn
function (c::MultipointConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    sys_syn::Subsystem{GABA_B_Synapse},
    sys_dst::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_syn.G * (sys_dst.V - sys_syn.E_syn)
end

# n1--->---------->n2
#          ↑
#         syn
function (c::MultipointConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    sys_syn::Subsystem{NMDA_Synapse},
    sys_dst::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    t)
    w = c.weight
    input = initialize_input(sys_dst)
    B = 1 / (1 + (exp(-(sys_dst.V + 30) / 4) * 1 / 3.57))
    @reset input.I_syn = -w * sys_syn.G * B * (sys_dst.V - sys_syn.E_syn)
end

# #-------------------------
# MSN receptors and modulators

function _neuron_voltage(sys::Subsystem{T}) where {T}
    states = GraphDynamics.get_states(sys)
    if :V_m in propertynames(states)
        sys.V_m
    elseif :V in propertynames(states)
        sys.V
    else
        _voltage_error(T)
    end
end

@noinline _voltage_error(T) = error("Subsystem{$T} has no voltage state V or V_m")

function _set_current_input(acc, value)
    if :I_syn in propertynames(acc)
        return @set acc.I_syn = value
    elseif :jcn in propertynames(acc)
        return @set acc.jcn = value
    elseif :I_app in propertynames(acc)
        return @set acc.I_app = value
    else
        _current_input_error()
    end
end

@noinline _current_input_error() = error("Input has no current port I_syn, jcn, or I_app")

function synapse_wiring_rule!(g,
    blox_src::AbstractNeuron,
    blox_syn::MsnNMDAR,
    blox_dst::AbstractNeuron; weight, kwargs...)
    add_node!(g, blox_src)
    add_node!(g, blox_syn)
    add_node!(g, blox_dst)
    @graph! g begin
        @connections begin
            # Provide both pre/post voltages to the synapse.
            blox_src => blox_syn, [conn = MultipointConnection(1.0, (; n_dst=PartitionedIndex(g, blox_dst)))]
            # Drive postsynaptic current using the synapse state.
            blox_src => blox_dst, [kwargs..., conn = MultipointConnection(weight, (; syn=PartitionedIndex(g, blox_syn)))]
        end
    end
end

function synapse_wiring_rule!(g,
    blox_src::Union{HHNeuronExci,HHNeuronInhib},
    blox_syn::MsnAMPAR,
    blox_dst::AbstractNeuron; weight, kwargs...)
    add_node!(g, blox_src)
    add_node!(g, blox_syn)
    add_node!(g, blox_dst)
    @graph! g begin
        @connections begin
            # Provide presynaptic drive to the synapse.
            blox_src => blox_syn, [conn = BasicConnection(1.0)]
            # Drive postsynaptic current using the synapse state.
            blox_src => blox_dst, [kwargs..., conn = MultipointConnection(weight, (; syn=PartitionedIndex(g, blox_syn)))]
        end
    end
end

function (c::MultipointConnection)(sys_src::Subsystem{<:AbstractNeuron},
    sys_dst::Subsystem{<:AbstractNeuron},
    sys_syn::Subsystem{MsnNMDAR},
    t)
    acc = initialize_input(sys_syn)
    @reset acc.V_pre = _neuron_voltage(sys_src)
    @reset acc.V_post = _neuron_voltage(sys_dst)
end

function (c::MultipointConnection)(sys_src::Subsystem{<:AbstractNeuron},
    sys_syn::Subsystem{MsnNMDAR},
    sys_dst::Subsystem{<:AbstractNeuron},
    t)
    v_post = _neuron_voltage(sys_dst)
    props = GraphDynamics.computed_properties_with_inputs(MsnNMDAR)
    input0 = initialize_input(sys_syn)
    input = merge(input0, (; V_post=v_post))
    i_nmda = props.I(sys_syn, input)
    acc = initialize_input(sys_dst)
    _set_current_input(acc, -c.weight * i_nmda)
end

function (c::BasicConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    sys_dst::Subsystem{MsnAMPAR},
    t)
    input = initialize_input(sys_dst)
    (; V, G_syn, V_shift, V_range) = sys_src
    @reset input.G_asymp = (G_syn / (1 + exp(-4.394 * ((V - V_shift) / V_range))))
end

function (c::MultipointConnection)(sys_src::Subsystem{<:Union{HHNeuronExci,HHNeuronInhib}},
    sys_syn::Subsystem{MsnAMPAR},
    sys_dst::Subsystem{<:AbstractNeuron},
    t)
    v_post = _neuron_voltage(sys_dst)
    acc = initialize_input(sys_dst)
    i_ampa = -c.weight * sys_syn.g * sys_syn.G * (v_post - sys_syn.E_syn)
    _set_current_input(acc, i_ampa)
end

function (c::BasicConnection)(sys_src::Subsystem{MsnD1Receptor},
    sys_dst::Subsystem{MsnNMDAR},
    t)
    acc = initialize_input(sys_dst)
    @reset acc.M_NMDA1 = c.weight * sys_src.M_NMDA1
end

function (c::BasicConnection)(sys_src::Subsystem{MsnD2Receptor},
    sys_dst::Subsystem{MsnAMPAR},
    t)
    acc = initialize_input(sys_dst)
    @reset acc.M_AMPA2 = c.weight * sys_src.M_AMPA2
end

function (c::BasicConnection)(sys_src::Subsystem{HTR5},
    sys_dst::Subsystem{BaxterSensoryNeuron},
    t)
    props = GraphDynamics.computed_properties_with_inputs(HTR5)
    input0 = initialize_input(sys_src)
    input = merge(input0, (; mode=sys_src.dummy))
    acc = initialize_input(sys_dst)
    merge(acc, (; PKA=c.weight * props.PKA(sys_src, input),
        PKC=c.weight * props.PKC(sys_src, input)))
end

function _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
    if g isa GraphSystem
        return _receptor_pair_wiring!(g.flat_graph, neuron, receptor; weight, kwargs...)
    end
    # Avoid duplicate nodes: add_node! appends even when names match, which can desync inputs from receptors.
    if !GraphDynamics.has_node(g, neuron)
        add_node!(g, neuron)
    end
    if !GraphDynamics.has_node(g, receptor)
        add_node!(g, receptor)
    end
    @graph! g begin
        @connections begin
            neuron => receptor, [conn = MultipointConnection(1.0, (; n_dst=PartitionedIndex(g, neuron)))]
            neuron => neuron, [kwargs..., conn = MultipointConnection(weight, (; syn=PartitionedIndex(g, receptor)))]
        end
    end
end

function GraphDynamics.system_wiring_rule!(g,
    neuron::TRNNeuron,
    receptor::Alpha7ERnAChR; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
    receptor::Alpha7ERnAChR,
    neuron::TRNNeuron; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
    neuron::TRNNeuron,
    receptor::CaTRPM4R; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
    receptor::CaTRPM4R,
    neuron::TRNNeuron; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function (c::MultipointConnection)(sys_src::Subsystem{TRNNeuron},
    sys_dst::Subsystem{TRNNeuron},
    sys_syn::Subsystem{Alpha7ERnAChR},
    t)
    acc = initialize_input(sys_syn)
    merge(acc, (; V_m=sys_dst.V_m, Ca_i=sys_dst.Ca_i))
end

function (c::MultipointConnection)(sys_src::Subsystem{TRNNeuron},
    sys_syn::Subsystem{Alpha7ERnAChR},
    sys_dst::Subsystem{TRNNeuron},
    t)
    acc = initialize_input(sys_dst)
    props = GraphDynamics.computed_properties_with_inputs(Alpha7ERnAChR)
    input0 = initialize_input(sys_syn)
    input = merge(input0, (; V_m=sys_dst.V_m, Ca_i=sys_dst.Ca_i))
    i_alpha7 = props.I_α7(sys_syn, input)
    j_er = props.J_ER(sys_syn, input)
    merge(acc, (; I_α7=c.weight * i_alpha7, J_ER=c.weight * j_er))
end

function (c::MultipointConnection)(sys_src::Subsystem{TRNNeuron},
    sys_dst::Subsystem{TRNNeuron},
    sys_syn::Subsystem{CaTRPM4R},
    t)
    acc = initialize_input(sys_syn)
    merge(acc, (; V=sys_dst.V_m))
end

function (c::MultipointConnection)(sys_src::Subsystem{TRNNeuron},
    sys_syn::Subsystem{CaTRPM4R},
    sys_dst::Subsystem{TRNNeuron},
    t)
    acc = initialize_input(sys_dst)
    i_trpm4 = sys_syn.ḡ * sys_syn.m * (sys_dst.V_m - sys_syn.E_CAN)
    merge(acc, (; I_app=-c.weight * i_trpm4))
end

# #-------------------------
# Muscarinic INCM/NCM module

function GraphDynamics.system_wiring_rule!(g,
    neuron::MuscarinicNeuron,
    receptor::MuscarinicR; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
    receptor::MuscarinicR,
    neuron::MuscarinicNeuron; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function (c::MultipointConnection)(sys_src::Subsystem{MuscarinicNeuron},
    sys_dst::Subsystem{MuscarinicNeuron},
    sys_syn::Subsystem{MuscarinicR},
    t)
    acc = initialize_input(sys_syn)
    merge(acc, (; V_m=sys_dst.V_m, Ca_i=sys_dst.Ca_i))
end

function (c::MultipointConnection)(sys_src::Subsystem{MuscarinicNeuron},
    sys_syn::Subsystem{MuscarinicR},
    sys_dst::Subsystem{MuscarinicNeuron},
    t)
    acc = initialize_input(sys_dst)
    props = GraphDynamics.computed_properties_with_inputs(MuscarinicR)
    input0 = initialize_input(sys_syn)
    input = merge(input0, (; V_m=sys_dst.V_m, Ca_i=sys_dst.Ca_i))
    i_ncm = props.I_NCM(sys_syn, input)
    i_nan = props.I_NaNCM(sys_syn, input)
    i_kn = props.I_KNCM(sys_syn, input)
    merge(acc, (; I_NCM=c.weight * i_ncm,
        I_NaNCM=c.weight * i_nan,
        I_KNCM=c.weight * i_kn))
end

# #-------------------------
# Beta2nAChR receptor module (Morozova et al. 2020)
# Connects to VTA DA and GABA neurons

function GraphDynamics.system_wiring_rule!(g,
    neuron::Union{VTADANeuron,VTAGABANeuron},
    receptor::Beta2nAChR; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
    receptor::Beta2nAChR,
    neuron::Union{VTADANeuron,VTAGABANeuron}; weight, kwargs...)
    _receptor_pair_wiring!(g, neuron, receptor; weight, kwargs...)
end

# Voltage from VTA neuron to receptor
function (c::MultipointConnection)(sys_src::Subsystem{<:Union{VTADANeuron,VTAGABANeuron}},
    sys_dst::Subsystem{<:Union{VTADANeuron,VTAGABANeuron}},
    sys_syn::Subsystem{Beta2nAChR},
    t)
    acc = initialize_input(sys_syn)
    merge(acc, (; V=sys_dst.V))
end

# Current from receptor to VTA neuron
function (c::MultipointConnection)(sys_src::Subsystem{<:Union{VTADANeuron,VTAGABANeuron}},
    sys_syn::Subsystem{Beta2nAChR},
    sys_dst::Subsystem{<:Union{VTADANeuron,VTAGABANeuron}},
    t)
    acc = initialize_input(sys_dst)
    props = GraphDynamics.computed_properties_with_inputs(Beta2nAChR)
    input0 = initialize_input(sys_syn)
    input = merge(input0, (; V=sys_dst.V))
    i_nachr = props.I(sys_syn, input)
    # I_ACh is outward positive. VTA neuron subtracts it: D(V) = (I_app - I_ionic - I_ACh)/C_m
    merge(acc, (; I_ACh=c.weight * i_nachr))
end
