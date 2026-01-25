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
                                             sys_view_src,
                                             neuron_src::Subsystem{LIF},
                                             foreach_connected_neuron) where {LIF <: LIFExciInhNeuron}
    t = integrator.t
    if t == neuron_src.t_refract_end # Refreactory period is over
        sys_view_src.is_refractory[] = 0
    else # Neuron fired
        # Begin refractory period
        sys_view_src.t_refract_end[] = t + neuron_src.t_refract_duration
        sys_view_src.is_refractory[] = 1
        
        add_tstop!(integrator, sys_view_src.t_refract_end[])

        # Reset the neuron voltage
        sys_view_src.V[] = neuron_src.V_reset

        # Now apply a function to each connected dst neuron
        foreach_connected_neuron() do conn, neuron_dst, sys_view_dst
            lif_exci_inh_update_connected_neuron(neuron_src, conn, neuron_dst, sys_view_src, sys_view_dst)
        end
    end
end
function lif_exci_inh_update_connected_neuron(neuron_src::Subsystem{LIFExciNeuron},
                                              conn::BasicConnection,
                                              neuron_dst::Subsystem{<:LIFExciInhNeuron},
                                              sys_view_src,
                                              sys_view_dst)
    w = conn.weight
    # check if the neuron is connected to itself
    if sys_view_src === sys_view_dst
        # x is the rise variable for NMDA synapses and it only applies to self-recurrent connections
        sys_view_dst.x[] += w
    end
    sys_view_dst.S_AMPA[] += w
    nothing
end
function lif_exci_inh_update_connected_neuron(neuron_src::Subsystem{LIFInhNeuron},
                                              conn::BasicConnection,
                                              neuron_dst::Subsystem{<:LIFExciInhNeuron},
                                              sys_view_src,
                                              sys_view_dst)
    w = conn.weight
    sys_view_dst.S_GABA[] += w
    nothing
end

function GraphDynamics.system_wiring_rule!(h,
                                           stim::PoissonSpikeTrain, 
                                           blox_dst::Union{LIFExciNeuron, LIFInhNeuron}; weight, kwargs...)
    spike_times = NeurobloxBasics.generate_spike_times(stim)
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
GraphDynamics.event_times((;t_spikes)::PoissonSpikeConn, sys_src, sys_dst) = t_spikes

GraphDynamics.has_discrete_events(::Type{PoissonSpikeTrain}) = true
function GraphDynamics.discrete_event_condition(p::Subsystem{PoissonSpikeTrain}, t, foreach_connected_neuron::F) where {F}
    # check if any of the downstream connections from p spike at time t.
    cond = mapreduce(|, foreach_connected_neuron; init=false) do conn, _, _
        t ∈ conn.t_spikes
    end
end
function GraphDynamics.apply_discrete_event!(integrator,
                                             sys_view_src,
                                             neuron_src::Subsystem{PoissonSpikeTrain},
                                             foreach_connected_neuron::F) where {F}
    t = integrator.t
    foreach_connected_neuron() do conn, neuron_dst, sys_view_dst
        # Check each downstream connection, if it's time to spike, increment the downstream neuron's S_AMPA_ext
        if t ∈ conn.t_spikes
            sys_view_dst.S_AMPA_ext[] += 1
        end
    end
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{LIFExciCircuit, LIFInhCircuit},
                                           blox_dst::Union{LIFExciCircuit, LIFInhCircuit}; kwargs...)
    neurons_src = nodes(blox_src.graph)
    neurons_dst = nodes(blox_dst.graph)
    for neuron_src ∈ neurons_src
        for neuron_dst ∈ neurons_dst
            system_wiring_rule!(g, neuron_src, neuron_dst; kwargs...)
        end
    end
end
function  GraphDynamics.system_wiring_rule!(g,
                                             stim::PoissonSpikeTrain,
                                             blox_dst::Union{LIFExciCircuit, LIFInhCircuit};
                                             kwargs...)
    neurons_dst = nodes(blox_dst.graph)
    for neuron_dst ∈ neurons_dst
        system_wiring_rule!(g, stim, neuron_dst; kwargs...)
    end
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


function GraphDynamics.system_wiring_rule!(g, blox_src::CanonicalMicroCircuit, blox_dest::CanonicalMicroCircuit; kwargs...)
    sysparts_src = get_parts(blox_src)
    sysparts_dest = get_parts(blox_dest)

    wm = get_weightmatrix(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dest))

    idxs = findall(!iszero, wm)
    for idx ∈ idxs
        system_wiring_rule!(g, sysparts_src[idx[2]], sysparts_dest[idx[1]]; kwargs..., weight=wm[idx])
    end
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractStimulus, blox_dest::CanonicalMicroCircuit; kwargs...)
    sysparts_dest = get_parts(blox_dest)
    system_wiring_rule!(g, blox_src, sysparts_dest[1]; kwargs...)
end
function GraphDynamics.system_wiring_rule!(g, blox_src::CanonicalMicroCircuit, blox_dest::AbstractObserver; kwargs...)
    sysparts_src = get_parts(blox_src)
    system_wiring_rule!(g, sysparts_src[2], blox_dest; kwargs...)
end

function (c::BasicConnection)(blox_src::Subsystem{<:NGNMM_QIF}, blox_dst::Subsystem{<:NGNMM_QIF}, t)
    w = c.weight
    x = output(blox_src)
    (; jcn = w * x)
end

function (c::BasicConnection)(sys_src::Subsystem{<:NGNMM_Izh}, sys_dest::Subsystem{<:NGNMM_Izh}, t)
    w = c.weight
    s_presyn = output(sys_src)
    (; jcn = w*sys_src.gₛ*s_presyn*(sys_dest.eᵣ-sys_dest.V))
end
