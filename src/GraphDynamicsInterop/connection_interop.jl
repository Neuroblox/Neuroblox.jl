##----------------------------------------------
## Connections
##----------------------------------------------

function define_basic_connection(c::Connector, blox_src::TSrc, blox_dst::TDst; mod=@__MODULE__()) where {TSrc, TDst}
    name_src = only(c.source)
    name_dst = only(c.destination)
    if isempty(c.weight)
        w = :_
    else
        w = only(c.weight)
    end 

    sys_src = get_namespaced_sys(blox_src)
    sys_dst = get_namespaced_sys(blox_dst)
    
    state_rules_src = map(ModelingToolkit.namespace_variables(sys_src)) do s
        @rule s => Expr(:., name_src, QuoteNode(Symbol(split(string(s.f), '₊')[end])))
    end
    param_rules_src = map(ModelingToolkit.namespace_parameters(sys_src)) do s
        @rule s => Expr(:., name_src, QuoteNode(Symbol(split(string(s), '₊')[end])))
    end
    state_rules_dst = map(ModelingToolkit.namespace_variables(sys_dst)) do s
        @rule s => Expr(:., name_dst, QuoteNode(Symbol(split(string(s.f), '₊')[end])))
    end
    param_rules_dst = map(ModelingToolkit.namespace_parameters(sys_dst)) do s
        @rule s => Expr(:., name_dst, QuoteNode(Symbol(split(string(s), '₊')[end])))
    end
    r = (Postwalk ∘ Chain)([[@rule w => Symbol(w)];
                            state_rules_src;
                            param_rules_src;
                            state_rules_dst;
                            param_rules_dst])

    nt = initialize_input(to_subsystem(blox_dst))
    length(c.equation) <= length(nt) || error("Too many equations for destination blox")
    eqs = map(keys(nt)) do lhs
        i = findfirst(c.equation) do eq
            Symbol(split(string(eq.lhs.f), "₊")[end]) == lhs
        end
        rhs = if isnothing(i)
            nt[lhs]
        else
            toexpr(r(c.equation[i].rhs))
        end
        Expr(:(=), lhs, rhs)
    end

    @eval mod begin
        function $GraphDynamics.system_wiring_rule!(h, blox_src::$TSrc, blox_dst::$TDst; weight, kwargs...)
            conn = BasicConnection(weight)
            add_connection!(g, blox_src, blox_dst; conn, kwargs..., weight, conn)
        end
        function (c::$BasicConnection)($name_src::$Subsystem{$TSrc}, $name_dst::$Subsystem{$TDst})
            $(Symbol(w)) = c.weight
            $(Expr(:tuple, eqs...))
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractBlox, blox_dst::AbstractBlox; weight, kwargs...)
    # The fallback for all blox: just use BasicConnection
    conn = BasicConnection(weight)
    if blox_src isa CompositeBlox || blox_dst isa CompositeBlox
        name_src = namespaced_nameof(blox_src)
        name_dst = namespaced_nameof(blox_dst)
        error("Tried to connect a composite blox using the fallback wiring rule, but this rule only works for non-composite blox. Source blox: $name_src, Destination blox: $name_dst")
    end
    add_connection!(g, blox_src, blox_dst; conn, weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{AbstractNeuronBlox, NeuralMassBlox},
                                           blox_dst::Union{AbstractNeuronBlox, NeuralMassBlox};
                                           weight, connection_rule="basic", kwargs...)
    conn = if connection_rule == "basic"
        BasicConnection(weight)
    elseif connection_rule == "psp"
        PSPConnection(weight)
    else
        ArgumentError("Unrecognized connection rule type, got $(connection_rule), expected either \"basic\" or \"psp\".")
    end
    add_connection!(g, blox_src, blox_dst; conn, weight, kwargs...)
end


##----------------------------------------------

struct BasicConnection{T} <: ConnectionRule
    weight::T
    BasicConnection{T}(x) where {T} = new{T}(x)
    BasicConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::BasicConnection, name_src, name_dst)
    (; weight = Symbol(:w_, name_src, :_, name_dst))
end
Base.zero(::Type{<:BasicConnection{T}}) where {T} = BasicConnection(zero(T))
Base.zero(::Type{BasicConnection}) = BasicConnection(0.0)

function (c::BasicConnection)(blox_src, blox_dst, t)
    (; jcn = c.weight * output(blox_src))
end

##----------------------------------------------

struct EventConnection{T, NT <: NamedTuple} <: ConnectionRule
    weight::T
    event_times::NT
    EventConnection(w::T, event_times::NT) where {T, NT} = new{float(T), NT}(w, event_times)
    EventConnection{T}(w, event_times::NT) where {T, NT} = new{T, NT}(w, event_times)
end
Base.zero(::Type{<:EventConnection{T}}) where {T} = EventConnection(zero(T), (;))
Base.zero(::Type{<:EventConnection}) = EventConnection(0.0, (;))

GraphDynamics.has_discrete_events(::EventConnection) = true
GraphDynamics.has_discrete_events(::Type{<:EventConnection{NT}}) where {NT} = true
function GraphDynamics.discrete_event_condition((;event_times)::EventConnection, t)
    t ∈ event_times
end
GraphDynamics.event_times((;event_times)::EventConnection) = event_times


##----------------------------------------------

struct ReverseConnection{T} <: ConnectionRule
    weight::T
    ReverseConnection{T}(x) where {T} = new{T}(x)
    ReverseConnection(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{ReverseConnection{T}}) where {T} = ReverseConnection(zero(T))
Base.zero(::Type{ReverseConnection}) = ReverseConnection(0.0)

##----------------------------------------------

struct PSPConnection{T} <: ConnectionRule
    weight::T
    PSPConnection{T}(x) where {T} = new{T}(x)
    PSPConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::PSPConnection, name_src, name_dst)
    (; weight = Symbol(:w_PSP_, name_src, :_, name_dst))
end
Base.zero(::Type{PSPConnection}) = PSPConnection(0.0)
Base.zero(::Type{PSPConnection{T}}) where {T} = PSPConnection(zero(T))

function (c::PSPConnection)(sys_src::Subsystem{<:AbstractNeuronBlox}, sys_dst::Subsystem{<:AbstractNeuronBlox}, t)
    (;jcn = c.weight * sys_src.G * (sys_src.E_syn - sys_dst.V))
end

##----------------------------------------------

struct HHConnection_STA{T} <: ConnectionRule
    weight::T
    HHConnection_STA{T}(x) where {T} = new{T}(x)
    HHConnection_STA(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::HHConnection_STA, name_src, name_dst)
    (; w = Symbol(:w_STA_, name_src, :_, name_dst))
end
Base.zero(::Type{HHConnection_STA{T}}) where {T} = HHConnection_STA(zero(T))
Base.zero(::Type{HHConnection_STA}) = HHConnection_STA(0.0)

##----------------------------------------------

struct HHConnection_GAP{T} <: ConnectionRule
    w_gap::T
    HHConnection_GAP{T}(x) where {T} = new{T}(x)
    HHConnection_GAP(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP}) = HHConnection_GAP(0.0)
Base.zero(::Type{HHConnection_GAP{T}}) where {T} = HHConnection_GAP(zero(T))

function GraphDynamics.connection_property_namemap(::HHConnection_GAP, name_src, name_dst)
    (; w_gap = Symbol(:w_GAP_, name_src, :_, name_dst))
end

##----------------------------------------------

struct HHConnection_GAP_Reverse{T} <: ConnectionRule
    w_gap_rev::T
    HHConnection_GAP_Reverse{T}(x) where {T} = new{T}(x)
    HHConnection_GAP_Reverse(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP_Reverse{T}}) where {T} = HHConnection_GAP_Reverse(zero(T))
Base.zero(::Type{HHConnection_GAP_Reverse}) = HHConnection_GAP_Reverse(0.0)

function GraphDynamics.connection_property_namemap(::HHConnection_GAP_Reverse, name_src, name_dst)
    (; w_gap_rev = Symbol(:w_GAP_reverse_, name_src, :_, name_dst))
end

##----------------------------------------------


function GraphDynamics.system_wiring_rule!(g, 
    HH_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    HH_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox};
    weight, sta=false, learning_rule=NoLearningRule(), kwargs...)
    
    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(HH_src.name, "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(HH_dst.name, "spikes_cumulative"))
    if sta & !(HH_src isa HHNeuronInhib_FSI_Adam_Blox) # Don't hit STA rules for FSI
        conn = HHConnection_STA(weight)
    else
        conn = BasicConnection(weight)
    end
    add_connection!(g, HH_src, HH_dst; conn, weight, learning_rule, kwargs...)
end

function (c::BasicConnection)(HH_src::Union{Subsystem{HHNeuronExciBlox},
                                            Subsystem{HHNeuronInhibBlox},
                                            Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                                            Subsystem{HHNeuronInhib_MSN_Adam_Blox},
                                            Subsystem{HHNeuronExci_STN_Adam_Blox},
                                            Subsystem{HHNeuronInhib_GPe_Adam_Blox}}, 
                              HH_dst::Union{Subsystem{HHNeuronExciBlox},
                                            Subsystem{HHNeuronInhibBlox},
                                            Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                                            Subsystem{HHNeuronInhib_MSN_Adam_Blox},
                                            Subsystem{HHNeuronExci_STN_Adam_Blox},
                                            Subsystem{HHNeuronInhib_GPe_Adam_Blox}},
                              t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.G * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end
function (c::HHConnection_STA)(HH_src::Union{Subsystem{HHNeuronExciBlox},
                                              Subsystem{HHNeuronInhibBlox},
                                              Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                                              Subsystem{HHNeuronInhib_MSN_Adam_Blox},
                                              Subsystem{HHNeuronExci_STN_Adam_Blox},
                                              Subsystem{HHNeuronInhib_GPe_Adam_Blox}}, 
                                HH_dst::Union{Subsystem{HHNeuronExciBlox},
                                              Subsystem{HHNeuronInhibBlox},
                                              Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                                              Subsystem{HHNeuronInhib_MSN_Adam_Blox},
                                              Subsystem{HHNeuronExci_STN_Adam_Blox},
                                              Subsystem{HHNeuronInhib_GPe_Adam_Blox}},
                               t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_dst.Gₛₜₚ * HH_src.G * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end
function (c::BasicConnection)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                              HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.Gₛ * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function GraphDynamics.system_wiring_rule!(g,
                                           HH_src::HHNeuronInhib_FSI_Adam_Blox, 
                                           HH_dst::HHNeuronInhib_FSI_Adam_Blox; weight, gap=false, kwargs...)

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


function ((;w_gap)::HHConnection_GAP)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, 
                                      HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap * (HH_dst.V - HH_src.V)
    acc
end

function ((;w_gap_rev)::HHConnection_GAP_Reverse)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, 
                                                  HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap_rev * (HH_dst.V - HH_src.V)
    acc
end

##----------------------------------------------
# Next Generation EI
function (c::BasicConnection)((;aₑ, bₑ, Cₑ)::Subsystem{NGNMM_theta}, 
                              HH_dst::Union{Subsystem{HHNeuronExciBlox}, Subsystem{HHNeuronInhibBlox}}, t)
    w = c.weight
    acc = initialize_input(HH_dst)
    a = aₑ
    b = bₑ
    C = Cₑ
    f = (1/(C*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)   
    @set acc.I_asc = w*f
end


#----------------------------------------------
# Kuramoto
function GraphDynamics.system_wiring_rule!(g, src::KuramotoOscillator, dst::KuramotoOscillator; weight, kwargs...)
    add_connection!(g, src, dst; weight, conn=BasicConnection(weight), kwargs...)
end

function (c::BasicConnection)(src::Subsystem{<:KuramotoOscillator},
                              dst::Subsystem{<:KuramotoOscillator}, t)
    w = c.weight
    x₀ = src.θ
    xᵢ = dst.θ
    (;jcn = w * sin(x₀ - xᵢ))
end

#----------------------------------------------
# LIFExci / LIFInh

function (c::BasicConnection)(sys_src::Subsystem{LIFExciNeuron},
                              sys_dst::Subsystem{<:Union{LIFExciNeuron, LIFInhNeuron}}, t)
    w = c.weight
    acc = initialize_input(sys_dst)
    @set acc.jcn = w * sys_src.S_NMDA * sys_dst.g_NMDA * (sys_dst.V - sys_dst.V_E) / (1 + sys_dst.Mg * exp(-0.062 * sys_dst.V) / 3.57)
end

function (c::BasicConnection)(sys_src::Subsystem{LIFInhNeuron},
                              sys_dst::Union{Subsystem{LIFExciNeuron}, Subsystem{LIFInhNeuron}}, t)
    initialize_input(sys_dst)
end

const LIFExciInhNeuron = Union{LIFExciNeuron, LIFInhNeuron}
GraphDynamics.has_discrete_events(::Type{LIFExciNeuron}) = true
GraphDynamics.has_discrete_events(::Type{LIFInhNeuron}) = true
function GraphDynamics.discrete_event_condition((; t_refract_end, V, θ)::Subsystem{LIF}, t, _) where {LIF <: LIFExciInhNeuron}
    # Triggers when either a refractory period is ending, or the neuron spiked (voltage exceeds threshold θ)
    (V > θ) || (t_refract_end == t)
end
function GraphDynamics.apply_discrete_event!(integrator,
                                             states_view_src, params_view_src,
                                             neuron_src::Subsystem{LIF},
                                             foreach_connected_neuron) where {LIF <: LIFExciInhNeuron}
    t = integrator.t
    if t == neuron_src.t_refract_end # Refreactory period is over
        params = params_view_src[]
        params_view_src[] = @set params.is_refractory = 0
    else # Neuron fired
        # Begin refractory period
        params_src = params_view_src[]
        @reset params_src.t_refract_end = t + params_src.t_refract_duration
        @reset params_src.is_refractory = 1
        
        add_tstop!(integrator, params_src.t_refract_end)
        params_view_src[] = params_src

        # Reset the neuron voltage
        states_view_src[:V] = params_src.V_reset

        # Now apply a function to each connected dst neuron
        foreach_connected_neuron() do conn, neuron_dst, states_view_dst, params_view_dst
            lif_exci_inh_update_connected_neuron(neuron_src, states_view_src, conn, neuron_dst, states_view_dst)
        end
    end
end
function lif_exci_inh_update_connected_neuron(neuron_src::Subsystem{LIFExciNeuron},
                                              states_view_src,
                                              conn::BasicConnection,
                                              neuron_dst::Subsystem{<:LIFExciInhNeuron},
                                              states_view_dst)
    w = conn.weight
    # check if the neuron is connected to itself
    if states_view_src === states_view_dst
        # x is the rise variable for NMDA synapses and it only applies to self-recurrent connections
        states_view_dst[:x] += w
    end
    states_view_dst[:S_AMPA] += w
    nothing
end
function lif_exci_inh_update_connected_neuron(neuron_src::Subsystem{LIFInhNeuron},
                                              states_view_src,
                                              conn::BasicConnection,
                                              neuron_dst::Subsystem{<:LIFExciInhNeuron},
                                              states_view_dst)
    w = conn.weight
    states_view_dst[:S_GABA] += w
    nothing
end



function GraphDynamics.system_wiring_rule!(h,
                                           stim::PoissonSpikeTrain, 
                                           blox_dst::Union{LIFExciNeuron, LIFInhNeuron}; weight, kwargs...)
    spike_times = Neuroblox.generate_spike_times(stim)
    conn = PoissonSpikeConn(weight, Set(spike_times))
    add_connection!(h, stim, blox_dst; kwargs..., conn)
end

struct PoissonSpikeConn{T} <: ConnectionRule
    w::T
    t_spikes::Set{Float64}
    PoissonSpikeConn{T}(x, t_spikes) where {T} = new{T}(x, t_spikes)
    PoissonSpikeConn(x::T, t_spikes) where {T} = new{float(T)}(x, t_spikes)
end
Base.zero(::Type{PoissonSpikeConn}) = PoissonSpikeConn(0.0, Set{Float64}())
Base.zero(::Type{PoissonSpikeConn{T}}) where {T} = PoissonSpikeConn(zero(T), Set{Float64}())
function ((;w)::PoissonSpikeConn)(stim::Subsystem{PoissonSpikeTrain},
                                  blox_dst::Union{Subsystem{LIFExciNeuron}, Subsystem{LIFInhNeuron}}, t)
    initialize_input(blox_dst)
end
GraphDynamics.event_times((;t_spikes)::PoissonSpikeConn) = (t_spikes)

GraphDynamics.has_discrete_events(::Type{PoissonSpikeTrain}) = true
function GraphDynamics.discrete_event_condition(p::Subsystem{PoissonSpikeTrain}, t, foreach_connected_neuron::F) where {F}
    # check if any of the downstream connections from p spike at time t.
    cond = mapreduce(|, foreach_connected_neuron; init=false) do conn, _, _, _
        t ∈ conn.t_spikes
    end
end
function GraphDynamics.apply_discrete_event!(integrator,
                                             states_view_src, params_view_src,
                                             neuron_src::Subsystem{PoissonSpikeTrain},
                                             foreach_connected_neuron::F) where {F}
    t = integrator.t
    foreach_connected_neuron() do conn, neuron_dst, states_view_dst, params_view_dst
        # Check each downstream connection, if it's time to spike, increment the downstream neuron's S_AMPA_ext
        if t ∈ conn.t_spikes
            states_view_dst[:S_AMPA_ext] += 1
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, blox::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}; kwargs...)
    neurons = blox.parts
    for n_src ∈ neurons
        for n_dst ∈ neurons
            system_wiring_rule!(g, n_src, n_dst; blox.kwargs...)
        end
    end
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{LIFExciCircuitBlox, LIFInhCircuitBlox},
                                           blox_dst::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}; kwargs...)
    neurons_src = blox_src.parts
    neurons_dst = blox_dst.parts
    for neuron_src ∈ neurons_src
        for neuron_dst ∈ neurons_dst
            system_wiring_rule!(g, neuron_src, neuron_dst; kwargs...)
        end
    end
end
function  GraphDynamics.system_wiring_rule!(g,
                                             stim::PoissonSpikeTrain,
                                             blox_dst::Union{LIFExciCircuitBlox, LIFInhCircuitBlox};
                                             kwargs...)
    neurons_dst = blox_dst.parts
    for neuron_dst ∈ neurons_dst
        system_wiring_rule!(g, stim, neuron_dst; kwargs...)
    end
end


##----------------------------------------------
# WinnerTakeAllBlox

function GraphDynamics.system_wiring_rule!(g, wta::WinnerTakeAllBlox; kwargs...)
    inh = wta.parts[1]
    for exci ∈ wta.parts[2:end]
        system_wiring_rule!(g, inh, exci; weight = 1.0)
        system_wiring_rule!(g, exci, inh; weight = 1.0)
    end
end

function GraphDynamics.system_wiring_rule!(g, wta_src::WinnerTakeAllBlox, wta_dst::WinnerTakeAllBlox; kwargs...)
    neurons_dst = get_exci_neurons(wta_dst)
    neurons_src = get_exci_neurons(wta_src)
    connection_matrix = get_connection_matrix(
        kwargs,
        namespaced_nameof(wta_src), namespaced_nameof(wta_dst),
        length(neurons_src), length(neurons_dst)
    )
    for (j, neuron_postsyn) in enumerate(neurons_dst)
        name_postsyn = namespaced_nameof(neuron_postsyn)
        for (i, neuron_presyn) in enumerate(neurons_src)
            name_presyn = namespaced_nameof(neuron_presyn)
            # Check names to avoid recurrent connections between the same neuron
            if (name_postsyn != name_presyn) && connection_matrix[i, j]
                system_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, neuron_src::HHNeuronInhibBlox, wta_dst::WinnerTakeAllBlox; kwargs...)
    neurons_dst = get_exci_neurons(wta_dst)
    for neuron_dst ∈ neurons_dst
        system_wiring_rule!(g, neuron_src, neuron_dst; kwargs...)
    end
end

##----------------------------------------------
# CorticalBlox
function GraphDynamics.system_wiring_rule!(g, c::CorticalBlox; kwargs...)
    wtas = c.parts[1:end-1]
    n_ff_inh = c.parts[end]
    system_wiring_rule!.((g,), wtas)
    system_wiring_rule!(g, n_ff_inh)
    (; connection_matrices) = c
    for i ∈ eachindex(wtas)
        for j ∈ eachindex(wtas)
            if i != j
                kwargs_ij = merge(NamedTuple(c.kwargs),
                                  (; connection_matrix = connection_matrices[i, j]))
                system_wiring_rule!(g, wtas[i], wtas[j]; kwargs_ij...)
            end
        end
        system_wiring_rule!(g, n_ff_inh, wtas[i]; weight = 1.0)
    end
end

function weight_matrix_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat =  get(kwargs, :weightmatrix) do
        error("Connection weight matrix from $name_src to $name_dst is not specified.")
    end
    if size(conn_mat) != (N_src, N_dst)
        error("The connection matrix must be of size $(N_src) x $(N_dst), got $(size(conn_mat))")
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if !iszero(conn_mat[i,j])
                system_wiring_rule!(g, neurons_src[i], neurons_dst[j]; kwargs..., weight=conn_mat[i,j])
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem,
                                           blox_src::Union{CorticalBlox,STN,Thalamus},
                                           blox_dst::Union{CorticalBlox,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    name_dst = namespaced_nameof(blox_dst)
    name_src = namespaced_nameof(blox_src)
    if haskey(kwargs, :weightmatrix)
        weight_matrix_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    else
        hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{CorticalBlox,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_inh_neurons(blox_src)
    name_dst = namespaced_nameof(blox_dst)
    name_src = namespaced_nameof(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::NGNMM_theta, blox_dst::CorticalBlox; kwargs...)
    n_ff_inh = blox_dst.parts[end]
    system_wiring_rule!(g, blox_src, n_ff_inh; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, s::Striatum; kwargs...)
    ns_inh = get_inh_neurons(s)
    matrisome = get_matrisome(s)
    striosome = get_striosome(s)
    system_wiring_rule!(g, matrisome)
    system_wiring_rule!(g, striosome)
    for inhib ∈ ns_inh
        system_wiring_rule!(g, inhib)
    end
end

function GraphDynamics.system_wiring_rule!(g, blox::Union{GPi, GPe}; kwargs...)
    for inhib ∈ get_inh_neurons(blox)
        system_wiring_rule!(g, inhib)
    end
end

function GraphDynamics.system_wiring_rule!(g, blox::STN; kwargs...)
    for inhib ∈ get_inh_neurons(blox)
        system_wiring_rule!(g, inhib)
    end
end

function GraphDynamics.system_wiring_rule!(g, blox::Thalamus; kwargs...)
    # connection_matrix=subcortical_connection_matrix(density, N_exci, weight)
    excis = get_exci_neurons(blox)
    for i ∈ eachindex(excis)
        system_wiring_rule!(g, excis[i])
        # for j ∈ eachindex(excis)
        #     cij = connection_matrix[i,j]
        #     if !iszero(cij)
        #         system_wiring_rule!(graph, excis[i], excis[j], weight=cij)
        #     end
        # end
    end
end

function subcortical_connection_matrix(density, N, weight)
    cm = zeros(N, N)
    for i ∈ 1:N
        for j ∈ 1:N
            if (i != j) && (rand() <= density) && cm[j,i] == 0
                cm[i,j] = weight
            end
        end
    end
    cm
end


#----------------------------------------------
# Discrete blox

function GraphDynamics.system_wiring_rule!(g::GraphSystem, ::AbstractActionSelection; kwargs...)
    #@info "Skipping the wiring of an ActionSelection"
    nothing
end
function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::AbstractBlox, ::AbstractActionSelection; kwargs...)
    # @info "Skipping the wiring of an ActionSelection"
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Striatum, sys_dst::Union{TAN, SNc}; kwargs...)
    system_wiring_rule!(g, get_striosome(sys_src), sys_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Striosome, sys_dst::Union{TAN,SNc};
                           weight, kwargs...)
    conn = BasicConnection(weight)
    add_connection!(g, sys_src, sys_dst; weight, conn, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{Striosome}, sys_dst::Subsystem{<:Union{TAN, SNc}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.H * sys_src.jcn_t_block
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem,
                                           sys_src::HHNeuronExciBlox,
                                           sys_dst::Union{Matrisome, Striosome};
                                           weight,
                                           learning_rule=NoLearningRule(),
                                           kwargs...)
    
    conn = BasicConnection(weight)
     learning_rule = maybe_set_state_pre( learning_rule, Symbol(namespaced_nameof(sys_src), :₊spikes_cumulative))
     learning_rule = maybe_set_state_post(learning_rule, Symbol(namespaced_nameof(sys_dst), :₊H_learning))
    add_connection!(g, sys_src, sys_dst; weight, conn, learning_rule, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{HHNeuronExciBlox}, sys_dst::Subsystem{<:Union{Matrisome, Striosome}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.spikes_window
end


function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Matrisome, sys_dst::Union{Matrisome, Striosome, HHNeuronInhibBlox}; weight=1.0, t_event, kwargs...)
    conn = EventConnection(weight, (;t_init=0.1, t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end

function (c::EventConnection)(src::Subsystem{Matrisome}, dst::Subsystem{<:Union{Matrisome, Striosome, HHNeuronInhibBlox}}, t)
    initialize_input(dst)
end


function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Matrisome})
    (;params_partitioned, partition_plan, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = partitioned(u, partition_plan)

    (;t_event) = ec.event_times
            
    params_dst = vparams_dst[]
    if haskey(ec.event_times, :t_init) && t == ec.event_times.t_init
        @reset params_dst.H = 1
    end
    if t == t_event
        @reset params_dst.H = m_src.ρ > m_dst.ρ ? 0 : 1
    end
    vparams_dst[] = params_dst
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
                    found = !iszero(M[l, j]) && !iszero(M[j, l])
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
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{HHNeuronInhibBlox})
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

function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Striosome})
    t = integrator.t
    (;t_init, t_event) = ec.event_times
    params_dst = vparams_dst[]
    if t == t_init
        @reset params_dst.H = 1
    else
        m_comp = find_competitor_matrisome(integrator, m_src, only(vparams_src.indices))
        if !isnothing(m_comp)
            @reset params_dst.H = m_src.ρ > m_comp.ρ ? 0 : 1
        end
    end
    vparams_dst[] = params_dst
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::TAN, sys_dst::Matrisome; weight=1.0, t_event, kwargs...)
    conn = EventConnection(weight, (; t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end
function (c::EventConnection)(sys_src::Subsystem{TAN}, sys_dst::Subsystem{Matrisome}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_dst.TAN_spikes
end


function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             sys_src::Subsystem{TAN},
                                             sys_dst::Subsystem{Matrisome})

    params_dst = vparams_dst[]
    w = ec.weight
    vparams_dst[] = @reset params_dst.TAN_spikes = w * rand(sys_src.rng, Poisson(sys_src.R))
    nothing
end


#--------------------
# ImageStimulus

function GraphDynamics.system_wiring_rule!(g::GraphSystem, stim::ImageStimulus, neuron::Union{HHNeuronInhibBlox, HHNeuronExciBlox}; current_pixel, weight, kwargs...)
    add_connection!(g, stim, neuron; conn=StimConnection(weight, current_pixel), weight, kwargs...)
end

struct StimConnection <: ConnectionRule
    weight::Float64
    pixel_index::Int
end

function (c::StimConnection)(src::Subsystem{ImageStimulus},
                             dst::Subsystem{<:Union{HHNeuronExciBlox, HHNeuronInhibBlox}},
                             t)
    w = c.weight
    input = initialize_input(dst)
    @reset input.I_in = w * src.current_image[c.pixel_index]
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, src::ImageStimulus, dst::CorticalBlox; kwargs...)
    for n_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, n_dst; current_pixel=src.current_pixel, kwargs...)
        increment_pixel!(src)
    end
end

#--------------------
# StimulusBlox
function (c::BasicConnection)(src::Subsystem{<:StimulusBlox}, dst::Subsystem{<:Union{HHNeuronInhibBlox, HHNeuronExciBlox}}, t)
    w = c.weight
    input = initialize_input(dst)
    x = only(outputs(src))
    @reset input.I_in = w * x
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, src::StimulusBlox, dst::Thalamus; kwargs...) 
    for neuron_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, neuron_dst; kwargs...)
    end
end


#----------------------------------------------
# Striatum|GPi|GPe - CorticalBlox|STN|Thalamus

function GraphDynamics.system_wiring_rule!(g, cb_src::Union{CorticalBlox,STN,Thalamus}, cb_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inh_neurons(cb_dst)
    neurons_src = get_exci_neurons(cb_src)
    hypergeometric_connections!(g,
                                neurons_src, neurons_dst,
                                namespaced_nameof(cb_src), namespaced_nameof(cb_dst); kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_src = get_inh_neurons(blox_src)
    neurons_dst = get_inh_neurons(blox_dst)
    hypergeometric_connections!(g, neurons_src, neurons_dst,
                                namespaced_nameof(blox_src), namespaced_nameof(blox_dst); kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{CorticalBlox,STN,Thalamus}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inh_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst,
                                namespaced_nameof(blox_src), namespaced_nameof(blox_dst); kwargs...)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHNeuronExciBlox, blox_dst::Union{Striatum, GPi}; kwargs...)
    for neuron_dst ∈ get_inh_neurons(blox_dst)
        system_wiring_rule!(g, blox_src, neuron_dst; kwargs...)
    end
end

# Adapted from the version of hypergeometric_connections in src/blox/connections.jl
function hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; rng=default_rng(), kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_connects = density * length(neurons_dst) * length(neurons_src)
    if length(neurons_src) == 0 || length(neurons_dst) == 0
        error(ArgumentError("hypergeometric_connections requires non-zero source and destination populations, got length(neurons_src) = $(length(neurons_src)), length(neurons_dst) = $(length(neurons_dst)) "))
    end
    out_degree = Int(ceil(N_connects / length(neurons_src)))
    in_degree =  Int(ceil(N_connects / length(neurons_dst)))
    wt = get_weight(kwargs, name_src, name_dst)
    # @info "" name_src name_dst (; density, out_degree, in_degree, N_connects)
    outgoing_connections = zeros(Int, length(neurons_src))
    for neuron_postsyn in neurons_dst
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rng, rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_src[idx]
                system_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        else
            for i in idx
                system_wiring_rule!(g, neurons_src[i], neuron_postsyn; kwargs..., weight=wt[i])
            end
        end
        outgoing_connections[idx] .+= 1
    end
end

# Adapted from the version of indegree_constrained_connections in src/blox/connections.jl
function indegree_constrained_connections!(g,
                                           neurons_src, neurons_dst,
                                           name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = Neuroblox.get_density(kwargs, name_src, name_dst)
        Neuroblox.indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                system_wiring_rule!(g, neurons_src[i], neurons_dst[j]; kwargs...)
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, cb::CorticalBlox, str::Striatum; kwargs...)
    neurons_dst = get_inh_neurons(str)
    neurons_src = get_exci_neurons(cb)
    w = get(kwargs, :weight) do
        error("No connection weight specified between $(namespaced_nameof(blox_src)) and $(namespaced_nameof(blox_dst))")
    end
    rng = get(kwargs, :rng, default_rng())
    dist = Uniform(0, 1)
    wt_ar = 2*w*rand(rng, dist, length(neurons_src)) # generate a uniform distribution of weight with average value w
    kwargs = (kwargs..., weight=wt_ar)
    if haskey(kwargs, :learning_rule)
        lr = kwargs.learning_rule
        matr = get_matrisome(str)
        lr = maybe_set_state_post(lr, namespaced_name(namespaced_nameof(matr), "H_learning"))
        kwargs = (kwargs..., learning_rule=lr)
    end
    hypergeometric_connections!(g,
                                neurons_src, neurons_dst,
                                namespaced_nameof(cb), namespaced_nameof(str); kwargs...)

    algebraic_parts = (get_matrisome(str), get_striosome(str))
    for (i, neuron_presyn) ∈ enumerate(neurons_src)
        kwargs = (kwargs...,weight=wt_ar[i])
        for part ∈ algebraic_parts
            system_wiring_rule!(g, neuron_presyn, part; kwargs...)
        end
    end
end


function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Striatum, sys_dst::Striatum; kwargs...)
    t_event = get(kwargs, :t_event) do
        name_src = namespaced_nameof(sys_src)
        name_dst = namespaced_nameof(sys_dst)
        error("No `t_event` provided for the connection between $(name_src) and $(name_dst)")
    end
    matrisome_src = get_matrisome(sys_src)
    matrisome_dst = get_matrisome(sys_dst)
    
    striosome_src = get_striosome(sys_src)
    striosome_dst = get_striosome(sys_dst)
    system_wiring_rule!(g, matrisome_src, matrisome_dst; t_event=t_event +   √(eps(t_event)), kwargs...)
    system_wiring_rule!(g, matrisome_src, striosome_dst; t_event=t_event + 2*√(eps(t_event)), kwargs...)
    for inhib ∈ get_inh_neurons(sys_dst)
        system_wiring_rule!(g, matrisome_src, inhib; t_event=t_event+2*√(eps(t_event)), kwargs...)
    end
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::TAN, sys_dst::Striatum; kwargs...)
    matr = get_matrisome(sys_dst)
    system_wiring_rule!(g, sys_src, matr; kwargs...)
end


# #-------------------------
# PING Network
struct PINGConnection{T} <: ConnectionRule
    w::T
    V_E::T
    V_I::T
    PINGConnection{T}(w, V_E, V_I) where {T} = new{float(T)}(w, V_E, V_I)
    PINGConnection(w::T, V_E::U, V_I::V) where {T, U, V} = new{float(promote_type(T, U, V))}(w, V_E, V_I)
end
PINGConnection(w; V_E=0.0, V_I=-80.0) = PINGConnection(w, V_E, V_I)
Base.zero(::Type{PINGConnection}) = PINGConnection(0.0, 0.0, 0.0)
Base.zero(::Type{PINGConnection{T}}) where {T} = PINGConnection(zero(T), zero(T), zero(T))

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractPINGNeuron, blox_dst::AbstractPINGNeuron; weight, kwargs...)
    V_E = get(kwargs, :V_E, 0.0)
    V_I = get(kwargs, :V_I, -80.0)
    conn = PINGConnection(weight; V_E, V_I)
    add_connection!(g, blox_src, blox_dst; weight, kwargs..., conn)
end

function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronExci}, blox_dst::Subsystem{PINGNeuronInhib}, t)
    (; w, V_E) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_E - V))
end

function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronInhib}, blox_dst::Subsystem{<:AbstractPINGNeuron}, t)
    (; w, V_I) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_I - V))
end

##----------------------------------------------
# Amygdala circuit
function GraphDynamics.system_wiring_rule!(g, c::LateralAmygdalaBlox; kwargs...)
    wtas = get_wtas(c)
    ff_neurons = get_ff_inh_neurons(c)

    N_ff_neuron = length(ff_neurons)
    N_wta = Int(length(wtas) / N_ff_neuron)

    system_wiring_rule!.((g,), wtas)
    system_wiring_rule!.((g,), ff_neurons)

    for k in eachindex(ff_neurons)
        for i in Base.OneTo(N_wta)
            wta_i = Int(i+((k-1)*N_wta))
            for j in Base.OneTo(N_wta)
                wta_j = Int(j+((k-1)*N_wta))
                if j != i
                    if haskey(c.kwargs, :connection_matrices)
                        kwargs_ij = merge(c.kwargs, Dict(:connection_matrix => c.kwargs[:connection_matrices][i+((k-1)*N_wta), j+((k-1)*N_wta)]))
                    else
                        kwargs_ij = Dict(c.kwargs)
                    end
                    system_wiring_rule!(g, wtas[wta_i], wtas[wta_j]; kwargs_ij...)
                end
            end
            system_wiring_rule!(g, ff_neurons[k], wtas[wta_i]; weight = 1.0)
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, neuron_src::Union{HHNeuronInhibBlox, NGNMM_theta}, la_dst::LateralAmygdalaBlox; kwargs...)
    neurons_dst = get_inh_neurons(la_dst)
    num = get_ff_inh_num(kwargs, namespaced_nameof(la_dst))
    system_wiring_rule!(g, neuron_src, neurons_dst[end-num]; kwargs...) 
end

function GraphDynamics.system_wiring_rule!(g, la_src::LateralAmygdalaBlox, la_dst::LateralAmygdalaBlox; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)

    hypergeometric_connections!(g, neurons_src, neurons_dst, namespaced_nameof(la_src), namespaced_nameof(la_dst); kwargs...)
end

function (c::BasicConnection)(
    sys_src::Subsystem{NGNMM_theta}, 
    sys_dst::Subsystem{<:Union{HHNeuronExciBlox, HHNeuronInhibBlox}}, 
    t
)
    a = sys_src.aₑ
    b = sys_src.bₑ
    f = (1/(sys_src.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)   

    acc = initialize_input(sys_dst)
    @set acc.I_asc = w*f
    
    return acc
end

function GraphDynamics.system_wiring_rule!(g, blox_src::NGNMM_theta, neuron_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; weight, kwargs...)
    conn = BasicConnection(weight)
    add_connection!(g, blox_src, neuron_dst; conn, weight, kwargs...)  
end

>>>>>>> f88145a (add GD wiring rules for the Amygdala circuit)
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

function GraphDynamics.system_wiring_rule!(g,
                                           cb_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                                           cb_dst::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam}; kwargs...)
    neurons_src = cb_src.parts 
    neurons_dst = cb_dst.parts
    indegree_constrained_connections!(g, neurons_src, neurons_dst,
                                      namespaced_nameof(cb_src), namespaced_nameof(cb_dst); kwargs...)
end

# #-------------------------
# NMDA receptor 
function GraphDynamics.system_wiring_rule!(g,
                                        blox_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox}, 
                                        blox_dst::MoradiNMDAR;
                                        weight, reverse=false, kwargs...)
    conn = reverse ? ReverseConnection(weight) : BasicConnection(weight)
    
    add_connection!(g, blox_src, blox_dst; conn, weight, reverse, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
                                        blox_src::MoradiNMDAR, 
                                        blox_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox};
                                        weight, kwargs...)
    conn = BasicConnection(weight)
    
    add_connection!(g, blox_src, blox_dst; conn, weight, kwargs...)
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Union{Subsystem{HHNeuronExciBlox}, Subsystem{HHNeuronInhibBlox}}, sys_dst::Subsystem{MoradiNMDAR}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.jcn = sys_src.z
    
    return acc
end

function (c::GraphDynamicsInterop.ReverseConnection)(sys_src::Union{Subsystem{HHNeuronExciBlox}, Subsystem{HHNeuronInhibBlox}}, sys_dst::Subsystem{MoradiNMDAR}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.V = sys_src.V
    
    return acc
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{MoradiNMDAR}, sys_dst::Union{Subsystem{HHNeuronExciBlox}, Subsystem{HHNeuronInhibBlox}}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    
    # HACK : we use sys_dst.V here because in another connection rule above we defined :
    # sys_src.V = sys_dst.V
    # However what we really need is sys_src.V which is an input state and this is why we can not access it here.
    Mg = 1 / (1 + sys_src.Mg_O * exp(-sys_src.z * sys_src.δ * sys_src.F * sys_dst.V / (sys_src.R * sys_src.T)) / sys_src.IC_50)
    I = -(sys_src.B - sys_src.A) * (sys_src.g_VI + sys_src.g) * Mg * (sys_dst.V - sys_src.E)
    
    
    acc = @set acc.I_syn = c.weight * I
    
    return acc
end
