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
        function (c::$BasicConnection)($name_src::$Subsystem{$TSrc}, $name_dst::$Subsystem{$TDst})
            $(Symbol(w)) = c.weight
            $(Expr(:tuple, eqs...))
        end
        function $GraphDynamicsInterop.blox_wiring_rule!(h, blox_src::$TSrc, blox_dst::$TDst, v_src, v_dst, kwargs)
            i, j = $only(v_src), $only(v_dst)
            (; conn, names) = $get_connection(blox_src, blox_dst, kwargs)
            $add_edge!(h, i, j, $Dict(:conn => conn, :names => names))
        end
        function $GraphDynamicsInterop.get_connection(blox_src::$TSrc, blox_dst::$TDst, kwargs)
            (;w_val, name) = $generate_weight_param(blox_src, blox_dst, kwargs)
            conn = $BasicConnection(w_val)
            (;conn, names = [name,])
        end
    end
end


#----------------------------------------------
get_nameof(sys) = nameof()
function generate_weight_param(blox_src, blox_dst, kwargs)
    w = get(kwargs, :weight) do
        error(ArgumentError("Expected a `:weight` to be set in connection."))
    end
    if w isa Num
        w_val = getdefault(w)
        name = nameof(w)
    else
        w_val = w
        name_src = namespaced_nameof(blox_src)
        name_dst = namespaced_nameof(blox_dst)
        
        name = Symbol("w_$(name_src)_$(name_dst)")
    end
    (;w_val, name)
end
function generate_gap_weight_param(blox_src, blox_dst, kwargs)
    w = get(kwargs, :gap_weight) do
        error(ArgumentError("Expected a `:gap_weight` to be set in connection."))
    end
    if w isa Num
        w_val = getdefault(w)
        name = nameof(w)
    else
        w_val = w
        name = Symbol("g_w_$(nameof(blox_src.system))_$(nameof(blox_dst.system))")
    end
    (;w_val, name)
end

function blox_wiring_rule!(h,
                           blox_src,
                           blox_dst,
                           v_src, v_dst, kwargs)
    #this is the fallback method for non-composite blox, hence vi and vj should have only one element
    i, j = only(v_src), only(v_dst)
    # put!(checker_channel, (; h, blox_src, blox_dst, i, j))
    # check_right_inds(h, blox_src, blox_dst, i, j)
    (; conn, names) = data = get_connection(blox_src, blox_dst, kwargs)
    add_edge!(h, i, j, Dict(:conn => conn, :names => names))
end

blox_wiring_rule!(h, blox, v, kwargs) = nothing
get_connection(g::MetaDiGraph, e::Edge) = get_connection(g, src(e), dst(e))
get_connection(g::MetaDiGraph, i::Int, j::Int) = let kwargs = props(g, i, j)
    ni = get_neuron(g, i)
    nj = get_neuron(g, j)
    get_connection(ni, nj, kwargs)
end
outer_nameof(x) = []


function check_right_inds(h, blox, ind)  
    sys = to_subsystem(blox)
    sys_ind = props(h, ind)[:subsystem]
    if sys != sys_ind
        error("Subsystem at index $ind does not match supplied blox (namespaced_nameof(blox)), this likely indicates that a method of `blox_wiring_rule` did something incorrect.")
    end
end

function check_right_inds(h, blox_src, blox_dst, i_src, i_dst)
    check_right_inds(h, blox_src, i_src)
    check_right_inds(h, blox_dst, i_dst)
end

##----------------------------------------------
function get_connection(blox_src, blox_dst, kwargs)
    (;w_val, name) = generate_weight_param(blox_src, blox_dst, kwargs)
    r_name = get(kwargs, :connection_rule, "basic")
    conn = if r_name == "basic"
        BasicConnection(w_val)
    elseif r_name == "psp"
        PSPConnection(w_val)
    else
        ArgumentError("Unrecognized connection rule type, got $(String(s)), expected either \"basic\" or \"psp\".")
    end
    (;conn, names = [name,])
end

struct BasicConnection <: ConnectionRule
    weight::Float64
end
Base.zero(::BasicConnection) = Base.zero(BasicConnection)
Base.zero(::Type{<:BasicConnection}) = BasicConnection(0.0)
function (c::BasicConnection)(blox_src, blox_dst)
    (; jcn = c.weight * output(blox_src))
end

struct PSPConnection <: ConnectionRule
    weight::Float64
end
Base.zero(::Type{<:PSPConnection}) = PSPConnection(0.0)
function (c::PSPConnection)(sys_src::Subsystem{<:AbstractNeuronBlox}, sys_dst::Subsystem{<:AbstractNeuronBlox})
    (;jcn = c.weight * sys_src.G * (sys_src.E_syn - sys_dst.V))
end

##----------------------------------------------

function get_connection(
    HH_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    HH_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox},
    kwargs)
    (;w_val, name) = generate_weight_param(HH_src, HH_dst, kwargs)

    if haskey(kwargs, :learning_rule)
        error(ArgumentError("got a connection with `:learning_rule` set, this is not yet supported."))
    end
    # if HH_src isa HHNeuronInhibBlox #&& HH_dst isa HHNeuronInhibBlox
    #     @show w_val
    # end
    STA = get(kwargs, :sta, false) & !(HH_src isa HHNeuronInhib_FSI_Adam_Blox) # Don't hit STA rules for FSI
    conn = HHConnection{STA}(w_val)
    (;conn, names = [name,])
end

struct HHConnection{STA} <: ConnectionRule
    w::Float64
end
HHConnection(w) = HHConnection{false}(w) # default to no STA
Base.zero(::Type{HHConnection{STA}}) where {STA} = HHConnection{STA}(0.0)
Base.zero(::HHConnection{STA}) where {STA} = HHConnection{STA}(0.0)
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
                                  HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox})
    acc = initialize_input(HH_dst)
    I_syn = -c.w * HH_src.Gₛ * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function get_connection(
    HH_src::HHNeuronInhib_FSI_Adam_Blox, 
    HH_dst::HHNeuronInhib_FSI_Adam_Blox,
    kwargs)
    (;w_val, name) = generate_weight_param(HH_src, HH_dst, kwargs)
    if haskey(kwargs, :learning_rule)
        error(ArgumentError("got a connection with `:learning_rule` set, this is not yet supported."))
    end
    if get(kwargs, :gap, false)
        GAP = true
        w_gap, gap_name = generate_gap_weight_param(HH_src, HH_dst, kwargs)
        w_gap_rev = let w
            w = kwargs[:gap_weight_reverse]
            w isa Num ? getdefault(w) : w
        end
        # maybe this should just be w_val, w_gap + w_gap_rev
        conn = HHConnection_GAP(w_val, w_gap, w_gap_rev)
        names = [name, gap_name]
    else
        w_val
        conn = HHConnection(w_val)
        names = [name]
    end
    (;conn, names)
end

struct HHConnection_GAP <: ConnectionRule
    w::Float64
    w_gap::Float64
    w_gap_rev::Float64
end
Base.zero(::Type{HHConnection_GAP}) = HHConnection_GAP(0.0, 0.0, 0.0)
function ((;w, w_gap, w_gap_rev)::HHConnection_GAP)(HH_src::Subsystem{HHNeuronInhib_FSI_Adam_Blox}, 
                                                    HH_dst::Subsystem{HHNeuronInhib_FSI_Adam_Blox})
    acc = HHConnection(w)(HH_src, HH_dst)
    acc = @set acc.I_gap = -(w_gap + w_gap_rev) * (HH_dst.V - HH_src.V)
    acc
end

##----------------------------------------------
# Next Generation EI
function get_connection(asc_src::NGNMM_theta, 
                        HH_dst::Union{HHNeuronExciBlox, HHNeuronInhibBlox},
                        kwargs)
    sys_src = get_namespaced_sys(asc_src)
    sys_dst = get_namespaced_sys(HH_dst)
    (;w_val, name) = generate_weight_param(asc_src, HH_dst, kwargs)
    (; conn = BasicConnection(w_val), names = [name,])
end

function (c::BasicConnection)((;aₑ, bₑ, Cₑ)::Subsystem{NGNMM_theta}, 
                              HH_dst::Union{Subsystem{HHNeuronExciBlox}, Subsystem{HHNeuronInhibBlox}})
    w = c.weight
    acc = initialize_input(HH_dst)
    f = (1 - aₑ^2 - bₑ^2)/((1+ 2*aₑ + aₑ^2 + bₑ^2) * (Cₑ*π))
    @set acc.I_asc = w*f
end


#----------------------------------------------
# Kuramoto
function get_connection(src::KuramotoOscillator, dst::KuramotoOscillator, kwargs)
    (;w_val, name) = generate_weight_param(src, dst, kwargs)
    (;conn=BasicConnection(w_val), names=[name])
end

function (c::BasicConnection)(src::Subsystem{<:KuramotoOscillator},
                              dst::Subsystem{<:KuramotoOscillator})
    w = c.weight
    x₀ = src.θ
    xᵢ = dst.θ
    (;jcn = w * sin(x₀ - xᵢ))
end

#----------------------------------------------
# LIFExci / LIFInh

function (c::BasicConnection)(sys_src::Subsystem{LIFExciNeuron},
                              sys_dst::Union{Subsystem{LIFExciNeuron}, Subsystem{LIFInhNeuron}})
    w = c.weight

    (; jcn = w * sys_src.S_NMDA * sys_dst.g_NMDA * (sys_dst.V - sys_dst.V_E) / 
        (1 + sys_dst.Mg * exp(-0.062 * sys_dst.V) / 3.57))
end

function (c::BasicConnection)(::Subsystem{LIFInhNeuron},
                              ::Union{Subsystem{LIFExciNeuron}, Subsystem{LIFInhNeuron}})
    (; jcn = 0.0)
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



function blox_wiring_rule!(h,
                           stim::PoissonSpikeTrain, 
                           blox_dst::Union{LIFExciNeuron, LIFInhNeuron},
                           v_src, v_dst, kwargs)
    i, j = only(v_src), only(v_dst)
    (; w_val, name) = generate_weight_param(stim, blox_dst, kwargs)
    conn = PoissonSpikeConn(w_val, Set(Neuroblox.generate_spike_times(stim)))
    add_edge!(h, i, j, Dict(:conn => conn, :names => [name]))
end
struct PoissonSpikeConn <: ConnectionRule
    w::Float64
    t_spikes::Set{Float64}
end
Base.zero(::Type{PoissonSpikeConn}) = PoissonSpikeConn(0.0, Set{Float64}())
function ((;w)::PoissonSpikeConn)(stim::Subsystem{PoissonSpikeTrain},
                                  blox_dst::Union{Subsystem{LIFExciNeuron}, Subsystem{LIFInhNeuron}})
    (; jcn = 0.0)
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


components(blox::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}) = blox.parts

issupported(::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}) = true
function blox_wiring_rule!(g, blox::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}, v, kwargs)
    neurons = components(blox)
    for i ∈ eachindex(neurons)
        for j ∈ eachindex(neurons)
            blox_wiring_rule!(g, neurons[i], neurons[j], v[i], v[j], blox.kwargs)
        end
    end
end


function blox_wiring_rule!(h,
                           blox_src::Union{LIFExciCircuitBlox, LIFInhCircuitBlox},
                           blox_dst::Union{LIFExciCircuitBlox, LIFInhCircuitBlox},
                           v_src, v_dst, kwargs)
    neurons_src = components(blox_src)
    neurons_dst = components(blox_dst)
    for (i, neuron_src) ∈ enumerate(neurons_src)
        for (j, neuron_dst) ∈ enumerate(neurons_dst)
            blox_wiring_rule!(h, neuron_src, neuron_dst, v_src[i], v_dst[j], kwargs)
        end
    end
end
function blox_wiring_rule!(h,
                           stim::PoissonSpikeTrain,
                           blox_dst::Union{LIFExciCircuitBlox, LIFInhCircuitBlox},
                           v_src, v_dst, kwargs)
    neurons_dst = components(blox_dst)
    for (j, neuron_dst) ∈ enumerate(neurons_dst)
        blox_wiring_rule!(h, stim, neuron_dst, only(v_src), v_dst[j], kwargs)
    end
end



##----------------------------------------------
# WinnerTakeAllBlox

function components(wta::WinnerTakeAllBlox)
    wta.parts
end
outer_nameof(wta::WinnerTakeAllBlox) = split(String(namespaced_nameof(wta)), '₊')



function blox_wiring_rule!(h, wta_src::WinnerTakeAllBlox, wta_dst::WinnerTakeAllBlox,
                           v_src, v_dst,
                           kwargs)
    
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
                blox_wiring_rule!(h, neuron_presyn, neuron_postsyn, v_src[1+i], v_dst[1+j], kwargs)
            end
        end
    end
end

issupported(::WinnerTakeAllBlox) = true
function blox_wiring_rule!(h, wta::WinnerTakeAllBlox, v, kwargs)
    i_inh = v[1]
    inh = wta.parts[1]
    for (i, i_exci) ∈ @views enumerate(v[2:end])
        exci = wta.parts[1+i]
        blox_wiring_rule!(h, inh, exci, i_inh, i_exci, Dict(:weight => 1.0))
        blox_wiring_rule!(h, exci, inh, i_exci, i_inh, Dict(:weight => 1.0))
    end
end

function blox_wiring_rule!(h, neuron_src::HHNeuronInhibBlox, wta_dst::WinnerTakeAllBlox, v_src, v_dst, kwargs)
    i = only(v_src)
    neurons_dst = Neuroblox.get_exci_neurons(wta_dst)
    for (j, neuron_dst) ∈ enumerate(neurons_dst)
        # 1 + i because v_dst[1] is the inhib neuron
        blox_wiring_rule!(h, neuron_src, neuron_dst, only(v_src), v_dst[1+j], kwargs)
    end
end

##----------------------------------------------
# CorticalBlox
issupported(::CorticalBlox) = true
components(c::CorticalBlox) = c.parts
outer_nameof(c::CorticalBlox) = split(String(namespaced_nameof(c)), '₊')
function blox_wiring_rule!(h, c::CorticalBlox, v, kwargs)
    wtas = c.parts[1:end-1]
    n_ff_inh = c.parts[end]
    N_wta = length(wtas)
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
                blox_wiring_rule!(h, wtas[i], wtas[j], v[i], v[j], kwargs_ij)
            end
        end
        blox_wiring_rule!(h, n_ff_inh, wtas[i], v[end], v[i], Dict(:weight => 1.0))
    end
end
# function blox_wiring_rule!(h, wta_src::CorticalBlox, wta_dst::WinnerTakeAllBlox,
#                            v_src, v_dst,
#                            kwargs)
#     blox_wiring_rule!(h, wta_dst, v_dst)
#     blox_wiring_rule!(h, wta_src, v_src)
    
#     neurons_dst = Neuroblox.get_exci_neurons(wta_dst)
#     neurons_src = Neuroblox.get_exci_neurons(wta_src)
#     rng = get(kwargs, :rng, default_rng())
#     density = get(kwargs, :density) do
#         error("Connection density from $(nameof(wta_src)) to $(nameof(wta_dst)) is not specified.")
#     end
#     dist = Bernoulli(density)
 
#     for (j, neuron_postsyn) in enumerate(neurons_dst)
#         name_postsyn = namespaced_nameof(neuron_postsyn)
#         for (i, neuron_presyn) in enumerate(neurons_src)
#             name_presyn = namespaced_nameof(neuron_presyn)
#             # Check names to avoid recurrent connections between the same neuron
#             if (name_postsyn != name_presyn) && rand(rng, dist)
#                 w_name = Symbol("w_$(nameof(neuron_presyn.system))_$(nameof(neuron_postsyn.system))")
#                 (; conn) = get_connection(neuron_presyn, neuron_postsyn, kwargs)
#                 add_edge!(h, v_dst[j], v_src[i], Dict(:conn => conn, :names => [w_name]))
#             end
#         end
#     end
# end

#----------------------------------------------
# Striatum_MSN_Adam
issupported(::Striatum_MSN_Adam) = true
components(s::Striatum_MSN_Adam) = s.parts
function blox_wiring_rule!(h, s::Striatum_MSN_Adam, v, kwargs)
    n_inh = s.parts
    N_inhib = length(n_inh)
    connection_matrix = s.connection_matrix
    for j ∈ axes(connection_matrix, 1)
        for i ∈ axes(connection_matrix, 2)
            cji = connection_matrix[j,i]
            if !iszero(cji)
                blox_wiring_rule!(h, n_inh[j], n_inh[i], v[j], v[i], Dict(:weight => cji))
            end
        end
    end
end

#----------------------------------------------
# Striatum_FSI_Adam
issupported(::Striatum_FSI_Adam) = true
components(s::Striatum_FSI_Adam) = s.parts
function blox_wiring_rule!(h, s::Striatum_FSI_Adam, v, kwargs)
    n_inh = s.parts
    N_inhib = length(n_inh)
    connection_matrix = s.connection_matrix
    # Here I'm creating a sub-graph g, populating it with edges describing the weights and gap_weights
    # just like how its done in Neuroblox.jl
    g = MetaDiGraph()
    add_vertices!(g, N_inhib)
    for i ∈ axes(connection_matrix, 2)
        for j ∈ axes(connection_matrix, 1)
            cji = connection_matrix[j, i]
            if iszero(cji.weight) && iszero(cji.g_weight) 
                nothing
            elseif iszero(cji.g_weight) 
                add_edge!(g, j, i, Dict(:weight=>cji.weight))
            else
                add_edge!(g, j, i, Dict(:weight=>cji.weight,
                                        :gap => true,
                                        :gap_weight => cji.g_weight)) 
            end
        end
    end
    # then I add the reversed gap_connections to the graph
    add_gap_backedges!(g)
    # then I use that graph to wire up h
    for edg ∈ edges(g)
        i, j = src(edg), dst(edg)
        pij = props(g, i, j)
        blox_wiring_rule!(h, n_inh[i], n_inh[j], v[i], v[j], pij)
    end
end

# function blox_wiring_rule!(h,
#                            blox_src::HHNeuronInhib_FSI_Adam_Blox,
#                            blox_dst::HHNeuronInhib_FSI_Adam_Blox,
#                            v_src, v_dst, kwargs)
#     #this is the fallback method for non-composite blox, hence vi and vj should have only one element
#     i, j = only(v_src), only(v_dst)
#     check_right_inds(h, blox_src, blox_dst, i, j)
#     add_gap_backedges!(h, (Edge(i, j) for i ∈ i for j ∈ j))
#     kwargs′ = merge(kwargs, props(h, i, j))
#     (; conn, names) = data = get_connection(blox_src, blox_dst, kwargs′)
#     add_edge!(h, i, j, Dict(:conn => conn, :names => names))
# end


#----------------------------------------------
# GPe_Adam
issupported(::GPe_Adam) = true
components(gpe::GPe_Adam) = gpe.parts
function blox_wiring_rule!(h, gpe::GPe_Adam, v, kwargs)
    n_inh = gpe.parts
    N_inhib = length(n_inh)
    connection_matrix = gpe.connection_matrix
    for i ∈ axes(connection_matrix, 1)
        for j ∈ axes(connection_matrix, 2)
            cij = connection_matrix[i,j]
            if !iszero(cij)
                blox_wiring_rule!(h, n_inh[i], n_inh[j], v[i], v[j], Dict(:weight => cij))
            end
        end
    end
end

#----------------------------------------------
# STN_Adam
issupported(::STN_Adam) = true
components(stn::STN_Adam) = stn.parts
function blox_wiring_rule!(h, stn::STN_Adam, v, kwargs)
    n_inh = stn.parts
    N_inhib = length(n_inh)
    connection_matrix = stn.connection_matrix
    for j ∈ axes(connection_matrix, 1)
        for i ∈ axes(connection_matrix, 2)
            cji = connection_matrix[j,i]
            if !iszero(cji)
                blox_wiring_rule!(h, n_inh[j], n_inh[i], v[j], v[i], Dict(:weight => cji))
            end
        end
    end
end

#----------------------------------------------
# Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam - Striatum_MSN_Adam|Striatum_FSI_Adam|GPe_Adam|STN_Adam

function blox_wiring_rule!(h,
                           cb_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                           cb_dst::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam,STN_Adam},
                           v_src, v_dst, kwargs)
    neurons_src = cb_src.parts 
    neurons_dst = cb_dst.parts
    indegree_constrained_connections!(h, neurons_src, neurons_dst,
                                      namespaced_nameof(cb_src), namespaced_nameof(cb_dst),
                                      v_src, v_dst, kwargs)
end


#----------------------------------------------
# Striatum|GPi|GPe - CorticalBlox|STN|Thalamus

function blox_wiring_rule!(h, cb_src::Union{CorticalBlox,STN,Thalamus}, cb_dst::Union{GPi, GPe},
                           v_src, v_dst, kwargs)
    neurons_src = get_inh_neurons(cb_src)
    neurons_dst = get_inh_neurons(cb_dst)
    hypergeometric_connections!(h,
                                neurons_src, neurons_dst,
                                namespaced_nameof(cb_src), namespaced_nameof(cb_dst),
                                v_src, v_dst,
                                kwargs)
end

function hypergeometric_connections!(h,
                                     neurons_src, neurons_dst,
                                     name_src, name_dst,
                                     v_src, v_dst,
                                     kwargs)
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
            kwargs′ = length(w_val) == 1 ? kwargs : merge(kwargs, Dict(:weight => w_val[i]))
            blox_wiring_rule!(h, neurons_src[i], neuron_dst, v_src[i], v_dst[j], kwargs′)
        end
        outgoing_connections[idx] .+= 1
    end
end

function indegree_constrained_connections!(h,
                                           neurons_src, neurons_dst,
                                           name_src, name_dst,
                                           v_src, v_dst,
                                           kwargs)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = Neuroblox.get_density(kwargs, name_src, name_dst)
        Neuroblox.indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                blox_wiring_rule!(h, neurons_src[i], neurons_dst[j], v_src[i], v_dst[j], kwargs)
            end
        end
    end
end


#----------------------------------------------
# Matrisome - Matrisome
function blox_wiring_rule!(h,
                           discr_src::Matrisome,
                           discr_dst::Matrisome,
                           v_src, v_dst, kwargs)
    i, j = only(v_src), only(v_dst)
    conn = get_connection(blox_src, blox_dst, kwargs)
    add_edge!(h, i, j, Dict(:conn => conn))
end

function get_connection(discr_src::Matrisome, discr_dst::Matrisome, kwargs)
    t_event = get(kwargs, :t_event) do
        error("Time for the event that affects the connection from $(namespaced_nameof(discr_src)) to $(namespaced_nameof(discr_dst)) is not specified.")
    end
    MMConn(t_event)
end

struct MMConn{T} <: ConnectionRule
    t_event::T
end

GraphDynamics.has_discrete_events(::MMConn) = true
GraphDynamics.has_discrete_events(::Type{MMConn}) = true
function GraphDynamics.discrete_event_condition((;t_event)::MMConn, t)
    t == t_event + sqrt(eps(t_event))
end
GraphDynamics.event_times((;t_event)::MMConn) = t_event + sqrt(eps(t_event))
GraphDynamics.discrete_events_require_inputs(::MMConn) = true
GraphDynamics.discrete_events_require_inputs(::Type{MMConn}) = true
function GraphDynamics.apply_discrete_event!(integrator,
                                            _, vparams_src,
                                            _, vparams_dst,
                                            ::MMConn,
                                            m_src::Subsystem{Matrisome}, jcn_src,
                                            m_dst ::Subsystem{Matrisome}, jcn_dst)
    H = m_src.H * jcn_src > m_dst.H * jcn_dst ? 0 : 1
    vparams_dst[] = SubsystemParams{Matrisome}((; H, TAN_spikes = m_dst.TAN_spikes, jcn_ = m_dst.jcn_, H_ = m_dst.H_))
    nothing
end

#----------------------------------------------
# HHNeuronExiBlox - Matrisome|Striosome

function blox_wiring_rule!(h,
                           neuron::HHNeuronExciBlox,
                           discr::Union{Matrisome, Striosome},
                           v_src, v_dst, kwargs)
    i, j = only(v_src), only(v_dst)
    (;conn, names) = get_connection(neuron, discr, kwargs)
    add_edge!(h, i, j, Dict(:conn => conn, :names => names))
end

function get_connection(neuron::HHNeuronExciBlox, discr::Union{Matrisome, Striosome}, kwargs)
    (;w_val, name) = generate_weight_param(neuron, discr, kwargs)
    if haskey(kwargs, :learning_rule)
        error("Learning rules are not yet supported")
    end
    (; conn = BasicConnection(w_val), names = [name])
end

# struct HHE_MS_Conn
#     w::Float64
# end
# Base.zero(::Type{HHE_MS_Conn}) = HHE_MS_Conn(0.0)
function (c::BasicConnection)((;spikes_window)::Subsystem{HHNeuronExciBlox},
                                 dst::Union{Subsystem{Matrisome}, Subsystem{Striosome}})
    w = c.weight
    w * spikes_window
end

#----------------------------------------------
# TAN - Matrisome

function blox_wiring_rule!(h,
                           neuron::TAN,
                           discr::Matrisome,
                           v_src, v_dst, kwargs)
    i, j = only(v_src), only(v_dst)
    (;conn, names) = get_connection(neuron, discr, kwargs)
    add_edge!(h, i, j, Dict(:conn => conn, :name => names))
end

function get_connection(discr_src::TAN, discr_dst::Matrisome, kwargs)
    (;w_val, name) = generate_weight_param(discr_src, discr_dst, kwargs)
    if haskey(kwargs, :learning_rule)
        error("Learning rules are not yet supported")
    end
    t_event = get(kwargs, :t_event) do
        error("Time for the event that affects the connection from $(namespaced_nameof(discr_src)) to $(namespaced_nameof(discr_dst)) is not specified.")
    end
    (; conn = TAN_M_Conn(w_val, t_event), names=[name])
end

struct TAN_M_Conn <: ConnectionRule
    w::Float64
    t_event::Float64
end
Base.zero(::Type{TAN_M_Conn}) = TAN_M_Conn(0.0, NaN)
function ((;w)::TAN_M_Conn)(src::Subsystem{TAN}, dst::Subsystem{Matrisome})
    w * dst.TAN_spikes
end

GraphDynamics.has_discrete_events(::TAN_M_Conn) = true
GraphDynamics.has_discrete_events(::Type{TAN_M_Conn}) = true
function GraphDynamics.discrete_event_condition((;t_event)::TAN_M_Conn, t)
    t == t_event + sqrt(eps(t_event))
end
GraphDynamics.event_times((;t_event)::TAN_M_Conn) = t_event + sqrt(eps(t_event))

GraphDynamics.discrete_events_require_inputs(::TAN_M_Conn) = true
GraphDynamics.discrete_events_require_inputs(::Type{TAN_M_Conn}) = true
function GraphDynamics.apply_discrete_event!(integrator,
                                            _, vparams_src,
                                            _, vparams_dst,
                                            ::TAN_M_Conn,
                                            tan_src::Subsystem{TAN}, jcn_src,
                                            mat_dst::Subsystem{Matrisome}, jcn_dst)
    (;κ,) = tan_src
    (;H, jcn_, H_) = mat_dst
    R = min(κ/(jcn_src + sqrt(eps())), κ)
    params = vparams_dst[]
    vparams_dst[] = @set params.TAN_spikes = float(rand(Poisson(R)))
    nothing
end

#----------------------------------------------
# Striosome - TAN|SNc

function blox_wiring_rule!(h,
                           discr_src::Striosome,
                           discr_dst::Union{TAN, SNc},
                           v_src, v_dst, kwargs)
    i, j = only(v_dst), only(v_src)
    (;conn, names) = get_connection(discr_src, discr_dst, kwargs)
    add_edge!(h, i, j, Dict(:conn => conn, :name => names))
end

function get_connection(discr_src::Striosome,
                        discr_dst::Union{TAN, SNc}, kwargs)
    (;w_val, name) = generate_weight_param(discr_src, discr_dst, kwargs)
    if haskey(kwargs, :learning_rule)
        error("Learning rules are not yet supported")
    end
    (; conn = BasicConnection(w_val), names=[name])
end

# struct S_TSN_Conn
#     w::Float64
# end
function (c::BasicConnection)(sys_src::Subsystem{Striosome}, sys_dst::Union{Subsystem{TAN}, Subsystem{SNc}})
    (;H, jcn_ref) = sys_src
    w = c.weight
    w * H * jcn_ref[]
end
# We have to tell GraphDynamics to run Striosomes before TAN or SNc blox so that we're sure the Striosomes
# have their inputs already set, otherwise the above rule is nonsense
GraphDynamics.must_run_before(::Type{Striosome}, ::Type{<:Union{TAN, SNc}}) = true

#----------------------------------------------
# Striatum - Striatum
issupported(::Striatum) = true
components(sta::Striatum) = sta.parts
function blox_wiring_rule!(h, str::Striatum, v_src, kwargs)
    # no internal wiring
    nothing 
end


function blox_wiring_rule!(h, str_src::Striatum, str_dst::Striatum, v_src, v_dst, kwargs)
    t_event = get(kwargs, :t_event) do
        error("Time for the event that affects the connection from $(namespaced_nameof(discr_src)) to $(namespaced_nameof(discr_dst)) is not specified.")
    end
    function SS_ev_builder(index_map)
        (i_neuron, _)        = index_map[v_src[1]]
        (i_matr, j_matr_src) = index_map[v_src[end-1]]
        (i_stri, j_stri_src) = index_map[v_src[end]]

        j_neurons_dst            = map(x -> index_map[x][2], v_dst)
        (_, j_matr_dst) = index_map[v_dst[end-1]]
        (_, j_stri_dst) = index_map[v_dst[end]]
        Striatum_Striatum_Composite_Event{i_matr, i_stri, i_neuron}(
            j_matr_src,
            j_stri_src,
            j_matr_dst,
            j_stri_dst,
            j_neurons_dst,
            t_event
        )
    end
    function SS_init_ev_builder(index_map)
        (i_neuron, _)        = index_map[v_src[1]]
        (i_matr, j_matr_src) = index_map[v_src[end-1]]
        (i_stri, j_stri_src) = index_map[v_src[end]]

        j_neurons_dst            = map(x -> index_map[x][2], v_dst)
        (_, j_matr_dst) = index_map[v_dst[end-1]]
        (_, j_stri_dst) = index_map[v_dst[end]]
        Striatum_Striatum_Composite_Event_Init{i_matr, i_stri, i_neuron}(
            j_matr_src,
            j_stri_src,
            j_matr_dst,
            j_stri_dst,
            j_neurons_dst
        )
    end
    
    push!(h.composite_discrete_events_builder, SS_ev_builder, SS_init_ev_builder)   
end

struct Striatum_Striatum_Composite_Event{i_matr, i_stri, i_neuron}
    j_matr_src::Int
    j_stri_src::Int
    j_matr_dst::Int
    j_stri_dst::Int
    j_neurons_dst::Vector{Int}
    event_time::Float64
end


GraphDynamics.event_times(ev::Striatum_Striatum_Composite_Event) = ev.event_time
GraphDynamics.discrete_event_condition(_, _, _, ev::Striatum_Striatum_Composite_Event, t) = (t == ev.event_time)


function GraphDynamics.apply_discrete_event!(integrator,
                                            states::NTuple{Len, Any},
                                            params::NTuple{Len, Any},
                                            connection_matrices::ConnectionMatrices{NConn},
                                            t,
                                            ev::Striatum_Striatum_Composite_Event{i_matr, i_stri, i_neuron}) where {Len, NConn, i_matr, i_stri, i_neuron}
    (;j_matr_src, j_stri_src, j_matr_dst, j_stri_dst, j_neurons_dst) = ev
    matr_src = Subsystem(states[i_matr][j_matr_src], params[i_matr][j_matr_src])
    matr_dst = Subsystem(states[i_matr][j_matr_dst], params[i_matr][j_matr_dst])
    
    stri_src = Subsystem(states[i_stri][j_stri_src], params[i_stri][j_stri_src])
    stri_dst = Subsystem(states[i_stri][j_stri_dst], params[i_stri][j_stri_dst])

    matr_src_jcn = calculate_inputs(Val(i_matr), j_matr_src, states, params, connection_matrices)
    matr_dst_jcn = calculate_inputs(Val(i_matr), j_matr_dst, states, params, connection_matrices)

    ρ_matr_src = matr_src.H * matr_src_jcn
    ρ_matr_dst = matr_dst.H * matr_dst_jcn
    
    pmatr_dst = params[i_matr][j_matr_dst]
    params[i_matr][j_matr_dst] = @set pmatr_dst.H = ifelse(ρ_matr_src > ρ_matr_dst, 0, 1)

    pstri_dst = params[i_stri][j_stri_dst]
    params[i_stri][j_stri_dst] = @set pstri_dst.H = ifelse(ρ_matr_src > ρ_matr_dst, 0, 1)

    for j_neuron_dst ∈ j_neurons_dst
        neuron_params = params[i_neuron][j_neuron_dst]
        params[i_neuron][j_neuron_dst] = @set neuron_params.I_bg = ifelse(ρ_matr_src > ρ_matr_dst, -2.0, 0.0)
    end
end


struct Striatum_Striatum_Composite_Event_Init{i_matr, i_stri, i_neuron}
    j_matr_src::Int
    j_stri_src::Int
    j_matr_dst::Int
    j_stri_dst::Int
    j_neurons_dst::Vector{Int}
end
GraphDynamics.event_times(ev::Striatum_Striatum_Composite_Event_Init) = 0.1
GraphDynamics.discrete_event_condition(_,  _, _, ev::Striatum_Striatum_Composite_Event_Init, t) = t == 0.1

function GraphDynamics.apply_discrete_event!(integrator, states::NTuple{Len, Any},
                                            params::NTuple{Len, Any},
                                            _,
                                            t,
                                            ev::Striatum_Striatum_Composite_Event_Init{i_matr,
                                                                                       i_stri,
                                                                                       i_neuron}) where {Len, i_matr, i_stri, i_neuron}
    (;j_matr_src, j_stri_src, j_matr_dst, j_stri_dst, j_neurons_dst) = ev
    pmatr_dst = params[i_matr][j_matr_dst]
    params[i_matr][j_matr_dst] = @set pmatr_dst.H = 1

    pstri_dst = params[i_stri][j_stri_dst]
    params[i_stri][j_stri_dst] = @set pstri_dst.H = 1

    for j_neuron_dst ∈ j_neurons_dst
        neuron_params = params[i_neuron][j_neuron_dst]
        params[i_neuron][j_neuron_dst] = @set neuron_params.I_bg = 0.0
    end
end

# #-------------------------
# PING Network
struct PINGConnection <: ConnectionRule
    w::Float64
    V_E::Float64
    V_I::Float64
end
Base.zero(::Type{PINGConnection}) = PINGConnection(0.0, 0.0, 0.0)

function get_connection(blox_src::PINGNeuronExci, blox_dst::PINGNeuronInhib, kwargs)
    (;w_val, name) = generate_weight_param(blox_src, blox_dst, kwargs)
    V_E = get(kwargs, :V_E, 0.0)
    (; conn = PINGConnection(w_val, V_E, 0.0), names=[name])
end
function get_connection(blox_src::PINGNeuronInhib, blox_dst::AbstractPINGNeuron, kwargs)
    (;w_val, name) = generate_weight_param(blox_src, blox_dst, kwargs)
    V_I = get(kwargs, :V_I, 0.0)
    (; conn = PINGConnection(w_val, V_I, -80.0), names=[name])
end

function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronExci}, blox_dst::Subsystem{PINGNeuronInhib})
    (; w, V_E) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_E - V))
end

function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronInhib}, blox_dst::Subsystem{<:AbstractPINGNeuron})
    (; w, V_I) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_I - V))
end

