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

struct BasicConnection <: ConnectionRule
    weight::Float64
end
Base.zero(::Type{<:BasicConnection}) = BasicConnection(0.0)
function (c::BasicConnection)(blox_src, blox_dst, t)
    (; jcn = c.weight * output(blox_src))
end

struct ReverseConnection <: ConnectionRule
    weight::Float64
end

Base.zero(::Type{<:ReverseConnection}) = ReverseConnection(0.0)

struct PSPConnection <: ConnectionRule
    weight::Float64
end
Base.zero(::Type{<:PSPConnection}) = PSPConnection(0.0)
function (c::PSPConnection)(sys_src::Subsystem{<:AbstractNeuronBlox}, sys_dst::Subsystem{<:AbstractNeuronBlox}, t)
    (;jcn = c.weight * sys_src.G * (sys_src.E_syn - sys_dst.V))
end

##----------------------------------------------

function GraphDynamics.system_wiring_rule!(g, 
    HH_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    HH_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox};
    weight, sta=false, kwargs...)
    
    if haskey(kwargs, :learning_rule)
        error(ArgumentError("got a connection with `:learning_rule` set, this is not yet supported."))
    end
    STA = sta & !(HH_src isa HHNeuronInhib_FSI_Adam_Blox) # Don't hit STA rules for FSI
    conn = HHConnection{STA}(weight)
    add_connection!(g, HH_src, HH_dst; conn, weight, kwargs...)
end

struct HHConnection{STA} <: ConnectionRule
    w::Float64
end
HHConnection(w) = HHConnection{false}(w) # default to no STA
Base.zero(::Type{HHConnection{STA}}) where {STA} = HHConnection{STA}(0.0)
function (c::HHConnection{STA})(HH_src::Union{Subsystem{HHNeuronExciBlox},
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
                                              Subsystem{HHNeuronInhib_GPe_Adam_Blox}}) where {STA}
    acc = initialize_input(HH_dst)
    if STA
        I_syn = -c.w * HH_dst.Gₛₜₚ * HH_src.G * (HH_dst.V - HH_src.E_syn)
    else
        I_syn = -c.w * HH_src.G * (HH_dst.V - HH_src.E_syn)
    end
    @set acc.I_syn = I_syn
end
function (c::HHConnection{false})(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox},
                                  HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, t)
    acc = initialize_input(HH_dst)
    I_syn = -c.w * HH_src.Gₛ * (HH_dst.V - HH_src.E_syn)
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
    conn = HHConnection(weight)
    add_connection!(g, HH_src, HH_dst; conn, weight, gap, kwargs...)
end

struct HHConnection_GAP <: ConnectionRule
    w_gap::Float64
end
Base.zero(::Type{HHConnection_GAP}) = HHConnection_GAP(0.0, 0.0, 0.0)
function ((;w_gap)::HHConnection_GAP)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, 
                                      HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap * (HH_dst.V - HH_src.V)
    acc
end

struct HHConnection_GAP_Reverse <: ConnectionRule
    w_gap_rev::Float64
end
Base.zero(::Type{HHConnection_GAP_Reverse}) = HHConnection_GAP_Reverse(0.0)
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
    f = (1 - aₑ^2 - bₑ^2)/((1+ 2*aₑ + aₑ^2 + bₑ^2) * (Cₑ*π))
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

struct PoissonSpikeConn <: ConnectionRule
    w::Float64
    t_spikes::Set{Float64}
end
Base.zero(::Type{PoissonSpikeConn}) = PoissonSpikeConn(0.0, Set{Float64}())
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
    connection_matrix = get_connection_matrix(kwargs,
                                              namespaced_nameof(wta_src), namespaced_nameof(wta_dst),
                                              length(neurons_src), length(neurons_dst))
    for (j, neuron_postsyn) in enumerate(neurons_dst)
        name_postsyn = namespaced_nameof(neuron_postsyn)
        for (i, neuron_presyn) in enumerate(neurons_src)
            name_presyn = namespaced_nameof(neuron_presyn)
            # Check names to avoid recurrent connections between the same neuron
            if (name_postsyn != name_presyn) && connection_matrix[i, j]
                # do 1+i because v[1] is the inh neuron
                system_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, neuron_src::HHNeuronInhibBlox, wta_dst::WinnerTakeAllBlox; kwargs...)
    neurons_dst = Neuroblox.get_exci_neurons(wta_dst)
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
    for i ∈ eachindex(wtas)
        for j ∈ eachindex(wtas)
            if i != j
                # users can supply a matrix of connection matrices.
                # connection_matrices[i,j][k, l] determines if neuron k from wta i is connected to
                # neuron l from wta j.
                if haskey(c.kwargs, :connection_matrices)
                    kwargs_ij = merge(c.kwargs, Dict(:connection_matrix => c.kwargs[:connection_matrices][i, j]))
                else
                    kargs_ij = Dict(c.kwargs)
                end
                system_wiring_rule!(g, wtas[i], wtas[j]; kwargs_ij...)
            end
        end
        system_wiring_rule!(g, n_ff_inh, wtas[i]; weight = 1.0)
    end
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

function GraphDynamics.system_wiring_rule!(g,
                                           cb_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                                           cb_dst::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam}; kwargs...)
    neurons_src = cb_src.parts 
    neurons_dst = cb_dst.parts
    indegree_constrained_connections!(g, neurons_src, neurons_dst,
                                      namespaced_nameof(cb_src), namespaced_nameof(cb_dst); kwargs...)
end


#----------------------------------------------
# Striatum|GPi|GPe - CorticalBlox|STN|Thalamus

function GraphDynamics.system_wiring_rule!(g, cb_src::Union{CorticalBlox,STN,Thalamus}, cb_dst::Union{GPi, GPe}; kwargs...)
    neurons_src = get_inh_neurons(cb_src)
    neurons_dst = get_inh_neurons(cb_dst)
    hypergeometric_connections!(g,
                                neurons_src, neurons_dst,
                                namespaced_nameof(cb_src), namespaced_nameof(cb_dst); kwargs...)
end

function hypergeometric_connections!(g,
                                     neurons_src, neurons_dst,
                                     name_src, name_dst; kwargs...)
    density = get(kwargs, :density) do
        error("Connection density from $(name_src) to $(name_dst) is not specified.")
    end
    rng = get(kwargs, :rng, default_rng())
    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))
    (;w_val, name) = generate_weight_param(HH_src, HH_dst, kwargs)
    outgoing_connections = zeros(Int, length(neurons_out))
    for (j, neuron_dst) ∈ enumerate(neurons_dst)
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rng, rem, min(in_degree, length(rem)); replace=false)
        for i ∈ idx
            kwargs′ = length(w_val) == 1 ? kwargs : merge(kwargs, (; weight = w_val[i]))
            system_wiring_rule!(g, neurons_src[i], neuron_dst; kwargs′...)
        end
        outgoing_connections[idx] .+= 1
    end
end

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


# #-------------------------
# PING Network
struct PINGConnection <: ConnectionRule
    w::Float64
    V_E::Float64
    V_I::Float64
end
PINGConnection(w; V_E=0.0, V_I=-80.0) = PINGConnection(w, V_E, V_I)
Base.zero(::Type{PINGConnection}) = PINGConnection(0.0, 0.0, 0.0)

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

# #-------------------------
# NMDA receptor 
function GraphDynamics.system_wiring_rule!(g,
                                        blox_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox}, 
                                        blox_dst::MoradiNMDAR;
                                        weight, reverse=false, kwargs...)
    conn = reverse ? ReverseConnection(weight) : BasicConnection(weight)
    
    add_connection!(g, blox_src, blox_dst; conn, weight, reverse, kwargs...)
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
    I = -(sys_src.B - sys_src.A) * sys_src.g * Mg * (sys_dst.V - sys_src.E)
    
    acc = @set acc.I_syn = c.weight * I
    
    return acc
end
