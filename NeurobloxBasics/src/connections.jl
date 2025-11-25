
function Connector(
    blox_src::KuramotoOscillator, 
    blox_dest::KuramotoOscillator; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    xₒ = only(outputs(blox_src; namespaced=true))
    xᵢ = only(outputs(blox_dest; namespaced=true)) #needed because this is also the θ term of the block receiving the connection

    eq = sys_dest.jcn ~ w*sin(xₒ - xᵢ)
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::LIFExciNeuron, 
    blox_dest::Union{LIFExciNeuron, LIFInhNeuron}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = sys_dest.jcn ~ w * sys_src.S_NMDA * sys_dest.g_NMDA * (sys_dest.V - sys_dest.V_E) / 
                    (1 + sys_dest.Mg * exp(-0.062 * sys_dest.V) / 3.57)
    
    # Compare the unique namespaced names of both systems
    sa = if nameof(sys_src) == nameof(sys_dest)
        # x is the rise variable for NMDA synapses and it only applies to self-recurrent connections
        nameof(sys_src) => [(sys_dest.S_AMPA, w), (sys_dest.x, w)]
    else
        nameof(sys_src) => [(sys_dest.S_AMPA, w)]
    end

    return Connector(nameof(sys_src), nameof(sys_dest); equation = eq, weight = [w], spike_affects = Dict(sa))
end

function Connector(
    blox_src::LIFInhNeuron, 
    blox_dest::Union{LIFExciNeuron, LIFInhNeuron}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    sa = nameof(sys_src) => [(sys_dest.S_GABA, w)]

    return Connector(nameof(sys_src), nameof(sys_dest); weight = w, spike_affects = Dict(sa))
end

function Connector(
    stim::PoissonSpikeTrain, 
    neuron::Union{LIFExciNeuron, LIFInhNeuron};
    kwargs...
)
    sys_dest = get_namespaced_sys(neuron)

    sa = namespaced_nameof(stim) => [sys_dest.S_AMPA_ext ~ sys_dest.S_AMPA_ext + 1]
    return Connector(namespaced_nameof(stim), nameof(sys_dest); spike_affects = Dict(sa))
end

function Connector(
    blox_src::Union{LIFExciCircuit, LIFInhCircuit}, 
    blox_dest::Union{LIFExciCircuit, LIFInhCircuit};
    kwargs...
)   
    neurons_src = get_neurons(blox_src)
    neurons_dest = get_neurons(blox_dest)

    C = Vector{Connector}(undef, length(neurons_src)*length(neurons_dest))
    i = 1
    for neuron_out in neurons_src
        for neuron_in in neurons_dest
            C[i] = Connector(neuron_out, neuron_in; kwargs...)
            i += 1
        end
    end

    return reduce(merge!, C)
end

function Connector(
    stim::PoissonSpikeTrain, 
    blox_dest::Union{LIFExciCircuit, LIFInhCircuit};
    kwargs...
)
    neurons_dest = get_neurons(blox_dest)

    conn = mapreduce(merge!, neurons_dest) do neuron
        Connector(stim, neuron; kwargs...)
    end

    return conn
end

function Connector(
    blox_src::NGNMM_Izh, 
    blox_dest::NGNMM_Izh; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    s_presyn = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ w*sys_src.gₛ*s_presyn*(sys_dest.eᵣ-sys_dest.V)
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::NGNMM_QIF, 
    blox_dest::NGNMM_QIF; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    x = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ w*x
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::PINGNeuronExci, 
    blox_dest::PINGNeuronInhib; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    V_E = haskey(kwargs, :V_E) ? kwargs[:V_E] : 0.0

    s = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ w*s*(V_E - sys_dest.V)
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::PINGNeuronInhib, 
    blox_dest::AbstractPINGNeuron; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    V_I = haskey(kwargs, :V_I) ? kwargs[:V_I] : -80.0    

    s = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ w*s*(V_I - sys_dest.V)
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::CanonicalMicroCircuit,
    blox_dest::CanonicalMicroCircuit;
    kwargs...
)
    sysparts_src = get_parts(blox_src)
    sysparts_dest = get_parts(blox_dest)

    wm = get_weightmatrix(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dest))

    idxs = findall(!iszero, wm)

    conn = mapreduce(merge!, idxs) do idx
        Connector(sysparts_src[idx[2]], sysparts_dest[idx[1]]; weight=wm[idx])
    end

    return conn
end

function Connector(
    blox_src::AbstractStimulus,
    blox_dest::CanonicalMicroCircuit;
    kwargs...
)
    sysparts_dest = get_parts(blox_dest)
    conn = Connector(blox_src, sysparts_dest[1]; kwargs...)

    return conn
end

function Connector(
    blox_src::CanonicalMicroCircuit,
    blox_dest::AbstractObserver;
    kwargs...
)
    sysparts_src = get_parts(blox_src)
    conn = Connector(sysparts_src[2], blox_dest; kwargs...)

    return conn
end
