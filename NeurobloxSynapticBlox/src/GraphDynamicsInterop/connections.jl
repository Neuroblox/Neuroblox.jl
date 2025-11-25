#==========================================================================================
Direct connections between Exci/Inhi neurons are implemented with an intermediate syntaptic block by default

i.e.
#   HHExci => HHExci

turns into

#   HHExci => Glu_AMPA_Synapse
#             Glu_AMPA_Synapse => HHExci

and

#   HHInhi => HHExci

turns into

#   HHInhi => GABA_A_Synapse
#             GABA_A_Synapse => HHExci
==========================================================================================#
function GraphDynamics.add_connection!(g::GraphSystem, blox_src::Union{HHExci, HHInhi}, blox_dst::Union{HHExci, HHInhi}; kwargs...)
    system_wiring_rule!(g, blox_src, blox_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHExci, blox_dst::Union{HHExci, HHInhi};
                           learning_rule=NoLearningRule(), sta = false, kwargs...)
    weight = get_weight(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst))

    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(namespaced_nameof(blox_src), "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(namespaced_nameof(blox_dst), "spikes_cumulative"))
    if sta
        #============================
        STA synapses require both the pre-synaptic voltage to calculate z, and the post synaptic voltage
        to calculate zₛₜₚ. Hence we make one connection from the presynaptic neuron to the synapse, and a
        ReverseConnection from the postsynaptic neuron to the synapse.
        # [n_pre::HHExci] -->G_asymp_pre--> [syn::Glu_AMPA_STA_Synapse] --->I_syn--> [n_post::HHExci]
        #                                     ↑                                       /
        #                                      \------------G_asymp_post<------------/
        # Because of this, we musn't ever re-use a pre-existing synapse!
        ============================#
        syn = Glu_AMPA_STA_Synapse(;name=Symbol("$(namespaced_nameof(blox_src))_$(namespaced_nameof(blox_dst))_STA_synapse"),
                                   E_syn=blox_src.E_syn, τ₂=blox_src.τ)
        # add_connection!(g, blox_src, syn; kwargs..., conn=BasicConnection(1.0)) 
        # add_connection!(g, blox_dst, syn; kwargs..., conn=ReverseConnection(1.0))
        # add_connection!(g, syn, blox_dst; kwargs..., weight, conn=BasicConnection(weight), learning_rule)
        add_connection!(g, blox_src, syn; kwargs..., weight=1) # weight=1 marks this as a forward rule 
        add_connection!(g, blox_dst, syn; kwargs..., weight=2) # weight=2 marks this as a reverse rule
        add_connection!(g, syn, blox_dst; kwargs..., conn=BasicConnection(weight), learning_rule)
    else
        # Generate a synapse (or fetch a pre-existing one)
        syn = get_synapse!(blox_src; kwargs...)
        conn = BasicConnection(weight)
        if !has_connection(g, blox_src, syn) # If we're re-using a synapse, don't re-add the connection
            add_connection!(g, blox_src, syn; kwargs..., weight, conn) # Note: connection between src and syn is not learnable!
        end
        add_connection!(g, syn, blox_dst; kwargs..., weight, conn, learning_rule)
    end
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHInhi, blox_dst::Union{HHExci, HHInhi}; kwargs...)
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse!(blox_src; kwargs...)
    # wire up the src to syn and syn to dst
    if !has_connection(g, blox_src, syn)
        # If we're re-using a synapse, don't re-add the connection
        system_wiring_rule!(g, blox_src, syn; kwargs...)
    end
    system_wiring_rule!(g, syn, blox_dst; kwargs...)
    nothing
end

"""
    (::BasicConnection)(src::Subsystem{<:Union{HHExci, HHInhi}}, dst::Subsystem{<:AbstractSynapse})

This connection simply forwards a presynaptic neuron's voltage to a synaptic block. The synaptic block is then
able to use that presyntaptic voltage in its connections with a postsynaptic neuron
"""
function (::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{<:Union{Glu_AMPA_Synapse, GABA_A_Synapse}}, t)
    input = initialize_input(sys_dst)
    (; V, G_syn, V_shift, V_range) = sys_src
    @reset input.G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
end

function (::BasicConnection)(sys_src::Subsystem{HHInhi}, sys_dst::Subsystem{<:Union{GABA_A_Synapse}}, t)
    input = initialize_input(sys_dst)
    # @set input.V = sys_src.V
    (; V, G_syn, V_shift, V_range) = sys_src
    @reset input.G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
end

function (c::BasicConnection)(sys_src::Subsystem{<:Union{GABA_A_Synapse, Glu_AMPA_Synapse}},
                              sys_dst::Subsystem{<:Union{HHExci, HHInhi}},
                              t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_src.g * sys_src.G * (sys_dst.V - sys_src.E_syn)
end

function (c::BasicConnection)(sys_src::Subsystem{NGNMM_theta}, sys_dst::Subsystem{<:Union{HHExci, HHInhi}}, t)
    w = c.weight
    a = sys_src.aₑ
    b = sys_src.bₑ
    acc = initialize_input(sys_dst)
    @reset acc.I_asc = w * (1/(sys_src.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)
end

# function (c::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{Glu_AMPA_STA_Synapse}, t)
#     input = initialize_input(sys_dst)
#     (; V, G_syn, V_shift, V_range) = sys_src
#     G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
#     @reset input.G_asymp_pre = G_asymp
# end

# function (c::ReverseConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{Glu_AMPA_STA_Synapse}, t)
#     input = initialize_input(sys_dst)
#     (; V, G_syn, V_shift, V_range) = sys_src
#     G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
#     @reset input.G_asymp_post = G_asymp
# end

function (c::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{Glu_AMPA_STA_Synapse}, t)
    input = initialize_input(sys_dst)
    (; V, G_syn, V_shift, V_range) = sys_src
    G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
    if c.weight == 1
        @reset input.G_asymp_pre = G_asymp
    elseif c.weight == 2
        @reset input.G_asymp_post = G_asymp
    else
        error("Weight flag for connections leading to GLU_AMPA_STA_synapse must be either 1 (presynaptic) or 2 (postsynaptic)")
    end
end


function (c::BasicConnection)(sys_src::Subsystem{Glu_AMPA_STA_Synapse}, sys_dst::Subsystem{<:Union{HHExci, HHInhi}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_src.g * sys_src.Gₛₜₚ * sys_src.G * (sys_dst.V - sys_src.E_syn)
end


#---------------------------------------------------------------------
# WinnerTakeAll

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::WinnerTakeAll, blox_dst::WinnerTakeAll; kwargs...)
    # users can supply a :connection_matrix to the graph edge, where
    # connection_matrix[i, j] determines if neurons_src[i] is connected to neurons_src[j]
    connection_matrix = get_connection_matrix(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dst),
                                              length(blox_src.excis), length(blox_dst.excis))
    
    for (j, neuron_postsyn) in enumerate(blox_dst.excis)
        name_postsyn = namespaced_nameof(neuron_postsyn)
        for (i, neuron_presyn) in enumerate(blox_src.excis)
            if name_postsyn != namespaced_nameof(neuron_presyn) && connection_matrix[i, j]
                system_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHInhi, blox_dst::WinnerTakeAll; kwargs...)
    for neuron_postsyn in blox_dst.excis
        system_wiring_rule!(g, blox_src, neuron_postsyn; kwargs...)
    end
end

#---------------------------------------------------------------------
# Cortical Blox, STN, Thalamus

function GraphDynamics.system_wiring_rule!(g::GraphSystem,
                                           blox_src::Union{Cortical,STN,Thalamus},
                                           blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    if haskey(kwargs, :weightmatrix)
        weight_matrix_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst); kwargs...)
    else
        hypergeometric_connections!(g, neurons_src, neurons_dst, namespaced_nameof(blox_src), namespaced_nameof(blox_dst); kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_inhi_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::NGNMM_theta, blox_dst::Cortical; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    system_wiring_rule!(g, blox_src, blox_dst.n_ff_inh; kwargs...)
end

#---------------------------------------------------------------------
# GPi

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    neurons_src = get_inhi_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Cortical,STN,Thalamus}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHExci, blox_dst::Union{Striatum, GPi}; kwargs...)
    for neuron_dst ∈ get_inhi_neurons(blox_dst)
        system_wiring_rule!(g, blox_src, neuron_dst; kwargs...)
    end
end

#---------------------------------------------------------------------
# Striatum

function GraphDynamics.system_wiring_rule!(g::GraphSystem, cb::Cortical, str::Striatum; kwargs...)
    neurons_dst = get_inhi_neurons(str)
    neurons_src = get_exci_neurons(cb)
    
    w = get_weight(kwargs, cb.name, str.name)

    dist = Uniform(0, 1)
    wt_ar = 2*w*rand(dist, length(neurons_src)) # generate a uniform distribution of weight with average value w
    kwargs = (kwargs..., weight=wt_ar)
    if haskey(kwargs, :learning_rule)
        lr = kwargs.learning_rule
        matr = str.matrisome
        lr = maybe_set_state_post(lr, namespaced_name(namespaced_nameof(matr), "H_learning"))
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

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Striatum, sys_dst::Striatum; kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    system_wiring_rule!(g, sys_src.matrisome, sys_dst.matrisome; t_event=t_event +   √(eps(t_event)), kwargs...)
    system_wiring_rule!(g, sys_src.matrisome, sys_dst.striosome; t_event=t_event + 2*√(eps(t_event)), kwargs...)
    for inhib ∈ sys_dst.inhibs
        system_wiring_rule!(g, sys_src.matrisome, inhib; t_event=t_event+2*√(eps(t_event)), kwargs...)
    end
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::TAN, sys_dst::Striatum; kwargs...)
    system_wiring_rule!(g, sys_src, sys_dst.matrisome; kwargs...)
end

#---------------------------------------------------------------------
# Discrete blox

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Striatum, sys_dst::Union{TAN, SNc}; kwargs...)
    system_wiring_rule!(g, sys_src.striosome, sys_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::HHExci, sys_dst::Union{Matrisome, Striosome};
                           weight, learning_rule=NoLearningRule(), kwargs...)
    
    conn = BasicConnection(weight)
    learning_rule = maybe_set_state_pre( learning_rule, Symbol(namespaced_nameof(sys_src), :₊spikes_cumulative))
    learning_rule = maybe_set_state_post(learning_rule, Symbol(namespaced_nameof(sys_dst), :₊H_learning))
    add_connection!(g, sys_src, sys_dst; weight, conn, learning_rule, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{<:Union{Matrisome, Striosome}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.spikes_window
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Matrisome, sys_dst::HHInhi; weight=1.0, kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    conn = EventConnection(weight, (;t_init=0.1, t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end

function (c::EventConnection)(src::Subsystem{Matrisome}, dst::Subsystem{HHInhi}, t)
    initialize_input(dst)
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{HHInhi})
    t = integrator.t
    params_src = vparams_src[]
    params_dst = vparams_dst[]
    (;t_init, t_event) = ec.event_times
    if t == t_init
        # @info "M-I Init"
        vparams_dst[] = @reset params_dst.I_bg = 0.0
    elseif t == t_event
        # @info "M-I Event"
        m_comp = find_competitor_matrisome(integrator, m_src, only(vparams_src.indices))
        if !isnothing(m_comp)
            vparams_dst[] = @reset params_dst.I_bg = m_src.ρ > m_comp.ρ ? -2.0 : 0.0
        end
    else
        error("Invalid event time, this shouldn't be possible")
    end
    nothing
end

#--------------------
# ImageStimulus
function GraphDynamics.system_wiring_rule!(g::GraphSystem, stim::ImageStimulus, neuron::Union{HHInhi, HHExci}; current_pixel, weight, kwargs...)
    add_connection!(g, stim, neuron; conn=StimConnection(weight, current_pixel), weight, kwargs...)
end

struct StimConnection <: ConnectionRule
    weight::Float64
    pixel_index::Int
end

function (c::StimConnection)(src::Subsystem{ImageStimulus}, dst::Subsystem{<:Union{HHInhi, HHExci}}, t)
    w = c.weight
    input = initialize_input(dst)
    @reset input.I_in = w * src.current_image[c.pixel_index]
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, src::ImageStimulus, dst::Cortical; kwargs...)
    for n_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, n_dst; current_pixel=src.current_pixel, kwargs...)
        increment_pixel!(src)
    end
end
