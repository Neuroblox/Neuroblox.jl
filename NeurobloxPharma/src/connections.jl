function Connector(
    blox_src::HHNeuronExci, 
    blox_dest::Union{HHNeuronExci, HHNeuronInhib, HHNeuronFSI}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))
    maybe_set_state_pre!(lr, sys_src.spikes_cumulative)
    maybe_set_state_post!(lr, sys_dest.spikes_cumulative)
        
    STA = get_sta(kwargs, nameof(blox_src), nameof(blox_dest))
    eq = if STA
        sys_dest.I_syn ~ -w * sys_dest.Gₛₜₚ * sys_src.G * (sys_dest.V - sys_src.E_syn)
    else
        sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)
    end

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, learning_rule=Dict(w => lr))
end

function Connector(
    blox_src::Union{HHNeuronInhib, HHNeuronFSI}, 
    blox_dest::Union{HHNeuronExci, HHNeuronInhib, HHNeuronFSI}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
        
    eq = sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::HHNeuronFSI,
    blox_dest::HHNeuronFSI; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = sys_dest.I_syn ~ -w * sys_src.Gₛ * (sys_dest.V - sys_src.E_syn)

    GAP = get_gap(kwargs, nameof(blox_src), nameof(blox_dest))
    if GAP
        w_gap = generate_gap_weight_param(blox_src, blox_dest; kwargs...)
        eq2 = sys_dest.I_gap ~ -w_gap * (sys_dest.V - sys_src.V)
        eq3 = sys_src.I_gap ~ -w_gap * (sys_src.V - sys_dest.V)

        return Connector(nameof(sys_src), nameof(sys_dest); equation=[eq, eq2, eq3], weight=[w, w_gap])
    else
        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    end
end

function Connector(
    blox_src::NGNMM_theta, 
    blox_dest::Union{HHNeuronExci, HHNeuronInhib, HHNeuronFSI}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    a = sys_src.aₑ
    b = sys_src.bₑ
    f = (1/(sys_src.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)   
    eq = sys_dest.I_asc ~ w*f
        
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::PulsesInput,
    blox_dest::Union{AbstractExciNeuron,AbstractInhNeuron};
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(outputs(blox_src; namespaced=true))

    eq = sys_dest.I_in ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::PulsesInput,
    blox_dest::Thalamus;
    kwargs...)

    neurons = get_exci_neurons(blox_dest)

    conn = mapreduce(merge!, neurons) do neuron
        Connector(blox_src, neuron; kwargs...)
    end

    return conn    
end

function Connector(
    blox_src::NGNMM_theta, 
    blox_dest::Cortical; 
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)

    conn = Connector(blox_src, neurons_dest[end]; kwargs...)

    return conn
end

function Connector(
    blox_src::WinnerTakeAll, 
    blox_dest::WinnerTakeAll; 
    kwargs...)
    neurons_src = get_exci_neurons(blox_src)
    neurons_dest = get_exci_neurons(blox_dest)
    # users can supply a :connection_matrix to the graph edge, where
    # connection_matrix[i, j] determines if neurons_src[i] is connected to neurons_src[j]
    connection_matrix = get_connection_matrix(kwargs,
                                              namespaced_nameof(blox_src), namespaced_nameof(blox_dest),
                                              length(neurons_src), length(neurons_dest))
    C = Connector[]
    for (j, neuron_postsyn) in enumerate(neurons_dest)
        name_postsyn = namespaced_nameof(neuron_postsyn)
        for (i, neuron_presyn) in enumerate(neurons_src)
            name_presyn = namespaced_nameof(neuron_presyn)
            # Check names to avoid recurrent connections between the same neuron
            if (name_postsyn != name_presyn) && connection_matrix[i, j]
                push!(C, Connector(neuron_presyn, neuron_postsyn; kwargs...))
            end
        end
    end
    
    # Check isempty(C) for the case of no connection being made. 
    # Connections between WTA neurons can be probabilistic so it's possible that none happen.
    if isempty(C)
        return Connector(namespaced_nameof(blox_src), namespaced_nameof(blox_dest))
    else
        return reduce(merge!, C)
    end
end

function Connector(
    blox_src::HHNeuronInhib, 
    blox_dest::WinnerTakeAll; 
    kwargs...
)
    neurons_dest = get_exci_neurons(blox_dest)

    conn = mapreduce(merge!, neurons_dest) do neuron_postsyn
        Connector(blox_src, neuron_postsyn; kwargs...)
    end

    return conn
end

function Connector(
    blox_src::Union{Cortical,STN,Thalamus},
    blox_dst::Union{Cortical,STN,Thalamus};
    kwargs...
)
    neurons_dest = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)

    cr = get_connection_rule(kwargs, blox_src, blox_dst)

    if cr == :density
        conn = density_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dst); kwargs...)
    elseif cr == :weightmatrix
        conn = weight_matrix_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dst); kwargs...)
    else
        conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dst); kwargs...)
    end
    
    return conn
end
function Connector(
    blox_src::Union{Cortical,STN,Thalamus},
    blox_dest::Union{GPi, GPe};
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)
    neurons_src = get_exci_neurons(blox_src)

    conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{Striatum, GPi, GPe},
    blox_dest::Union{Cortical, STN, Thalamus};
    kwargs...
)
    neurons_dest = get_exci_neurons(blox_dest)
    neurons_src = get_inh_neurons(blox_src)

    conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{Striatum, GPi, GPe},
    blox_dest::Union{GPi, GPe};
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)
    neurons_src = get_inh_neurons(blox_src)

    conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    cb::Cortical,
    str::Striatum;
    kwargs...
)
    neurons_dest = get_inh_neurons(str)
    neurons_src = get_exci_neurons(cb)

    rng = get(kwargs, :rng, Random.default_rng())
    w = get_weight(kwargs, namespaced_nameof(cb), namespaced_nameof(str))
    dist = Uniform(0,1)
    wt_ar = 2*w*rand(rng, dist, length(neurons_src)) # generate a uniform distribution of weight with average value w 
    kwargs = (kwargs..., weight=wt_ar)

    if haskey(kwargs, :learning_rule)
        lr = get_learning_rule(kwargs, namespaced_nameof(cb), namespaced_nameof(str))
        sys_matr = get_namespaced_sys(get_matrisome(str))
        maybe_set_state_post!(lr, sys_matr.H_learning)
        kwargs = (kwargs..., learning_rule=lr)
    end

    conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(cb), nameof(str); kwargs...)

    algebraic_parts = [get_matrisome(str), get_striosome(str)]

    for (i,neuron_presyn) in enumerate(neurons_src)
        kwargs = (kwargs...,weight=wt_ar[i])
        for part in algebraic_parts
            merge!(conn, Connector(neuron_presyn, part; kwargs...))
        end
    end

    return conn
end

function Connector(
    neuron::HHNeuronExci,
    str::Union{Striatum, GPi};
    kwargs...
)
    neurons_dest = get_inh_neurons(str)
    neuron_src = neuron

    conn = mapreduce(merge!, neurons_dest) do neuron_dest
        Connector(neuron_src, neuron_dest; kwargs...)
    end
    
    return conn
end

function Connector(
    blox_src::HHNeuronExci,
    blox_dest::Union{Matrisome, Striosome};
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))
    maybe_set_state_pre!(lr, sys_src.spikes_cumulative)
    maybe_set_state_post!(lr, sys_dest.H_learning)


    eq = sys_dest.jcn ~ w*sys_src.spikes_window

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, learning_rule=Dict(w => lr))
end

function Connector(
    blox_src::Striatum,
    blox_dest::Striatum;
    kwargs... 
)
    sys_matr_src = get_namespaced_sys(get_matrisome(blox_src))
    sys_matr_dest = get_namespaced_sys(get_matrisome(blox_dest))
    sys_strios_dest = get_namespaced_sys(get_striosome(blox_dest))
    neurons_dest = get_inh_neurons(blox_dest)

    t_event = get_event_time(kwargs, nameof(blox_src), nameof(blox_dest))
    cb_matr = [t_event] => [sys_matr_dest.H ~ ifelse(sys_matr_src.H*sys_matr_src.jcn > sys_matr_dest.H*sys_matr_dest.jcn, 0, 1)]
    cb_strios = [t_event] => [sys_strios_dest.H ~ ifelse(sys_matr_src.H*sys_matr_src.jcn > sys_matr_dest.H*sys_matr_dest.jcn, 0, 1)]
    
    # HACK: H should be reset to 1 at the beginning of each trial
    # Such callbacks should be moved to RL-specific functions like `run_experiment!`
    cb_matr_init = [0.1] => [sys_matr_dest.H ~ 1]
    cb_strios_init = [0.1] => [sys_strios_dest.H ~ 1]

    dc = [cb_matr, cb_strios, cb_matr_init, cb_strios_init]

    for neuron in neurons_dest
        sys_neuron = get_namespaced_sys(neuron)
        # Large negative current added to shut down the Striatum spiking neurons.
        # Value is hardcoded for now, as it's more of a hack, not user option. 
        cb_neuron = [t_event] => [sys_neuron.I_bg ~ ifelse(sys_matr_src.H*sys_matr_src.jcn > sys_matr_dest.H*sys_matr_dest.jcn, -2, 0)]
        # lateral inhibition current I_bg should be set to 0 at the beginning of each trial
        cb_neuron_init = [0.1] => [sys_neuron.I_bg ~ 0]
        push!(dc, cb_neuron)
        push!(dc, cb_neuron_init)
    end

    w = generate_weight_param(blox_src, blox_dest; weight=1)

    return Connector(namespaced_nameof(blox_src), namespaced_nameof(blox_dest); discrete_callbacks=dc, weight=w)
end

function Connector(
    blox_src::Striatum,
    blox_dest::Union{TAN, SNc};
    kwargs... 
)
    striosome = get_striosome(blox_src)
    return Connector(striosome, blox_dest; kwargs...)
end

function Connector(
    blox_src::Striosome,
    blox_dest::Union{TAN, SNc};
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = sys_dest.jcn ~ w*sys_src.H*sys_src.jcn

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, learning_rule=Dict(w => lr))
end

function Connector(
    blox_src::TAN,
    blox_dest::Striatum;
    kwargs...
) 
    matrisome = get_matrisome(blox_dest)
    
    return Connector(blox_src, matrisome; kwargs...)
end

sample_poisson(λ) = rand(Poisson(λ))
@register_symbolic sample_poisson(λ)

"""
    Non-symbolic, time-block-based way of `@register_symbolic sample_poisson(λ)`. 
"""
function sample_affect!(integ, u, p, ctx)
    R = min(integ.p[p[1]]/(integ.p[p[2]] + sqrt(eps())), integ.p[p[1]])
    v = rand(integ.p[p[4]], Poisson(R))
    integ.p[p[3]] = v
end

function Connector(
    blox_src::TAN,
    blox_dest::Matrisome;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    t_event = get_event_time(kwargs, nameof(blox_src), nameof(blox_dest))
    cb = [t_event+sqrt(eps(t_event))] => (sample_affect!, [], [sys_src.κ, sys_src.jcn, sys_dest.TAN_spikes, sys_src.rng], [])

    eq = sys_dest.jcn ~ w*sys_dest.TAN_spikes

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, discrete_callbacks=cb, learning_rule=Dict(w => lr))
end

function Connector(
    blox_src::Matrisome,
    blox_dest::Matrisome;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    t_event = get_event_time(kwargs, nameof(blox_src), nameof(blox_dest))
    cb = [t_event] => [sys_dest.H ~ ifelse(sys_src.H*sys_src.jcn > sys_dest.H*sys_dest.jcn, 0, 1)]

    return Connector(nameof(sys_src), nameof(sys_dest); discrete_callbacks=cb)
end

function Connector(
    stim::ImageStimulus,
    neuron::Union{HHNeuronExci, HHNeuronInhib};
    kwargs...
)   
    sys_src = get_namespaced_sys(stim)
    sys_dest = get_namespaced_sys(neuron)

    pixels = namespace_parameters(sys_src)

    w = generate_weight_param(stim, neuron; kwargs...)

    # No check for kwargs[:learning_rule] here. 
    # The connection from stimulus is conceptual, the weight can not be updated.

    eq = sys_dest.I_in ~ w * pixels[stim.current_pixel]
    
    increment_pixel!(stim)

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    stim::ImageStimulus,
    cb::Cortical;
    kwargs...
)
    neurons = get_exci_neurons(cb)

    conn = mapreduce(merge!, neurons) do neuron
        Connector(stim, neuron; kwargs...)
    end

    return conn
end

function connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, get_matrisome(str1), get_matrisome(str2))
end

function connect_action_selection!(as::AbstractActionSelection, matr1::Matrisome, matr2::Matrisome)
    sys1 = get_namespaced_sys(matr1)
    sys2 = get_namespaced_sys(matr2)

    as.competitor_states = [sys1.ρ_, sys2.ρ_] #HACK : accessing values of rho at a specific time after the simulation
end
