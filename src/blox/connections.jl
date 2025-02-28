struct Connector
    source::Vector{Symbol}
    destination::Vector{Symbol}
    equation::Vector{Equation}
    weight::Vector{Num}
    delay::Vector{Num}
    discrete_callbacks
    spike_affects::Dict{Symbol, Vector{Union{Tuple{Num, Num}, Equation}}}
    learning_rule::Dict{Num, AbstractLearningRule}
end

function Connector(
    src::Union{Symbol, Vector{Symbol}}, 
    dest::Union{Symbol, Vector{Symbol}}; 
    equation=Equation[], 
    weight=Num[], 
    delay=Num[], 
    discrete_callbacks=[], 
    spike_affects=Dict{Symbol, Vector{Tuple{Num, Num}}}(),
    learning_rule=Dict{Num, AbstractLearningRule}(),
    connection_blox=Set([])
    )
    filter!(x -> !isempty(last(x)), spike_affects)
    # Check if all weigths have NoLearningRule and if so don't keep them in the final Dict.
    U = narrowtype_union(learning_rule)
    learning_rule = U <: NoLearningRule ? Dict{Num, NoLearningRule}() : learning_rule

    Connector(
        to_vector(src), 
        to_vector(dest), 
        to_vector(equation), 
        to_vector(weight), 
        to_vector(delay), 
        to_vector(discrete_callbacks), 
        spike_affects, 
        learning_rule
    )
end

function Base.isempty(conn::Connector)
    return isempty(conn.equation) && isempty(conn.weight) && isempty(conn.delay) && isempty(conn.discrete_callbacks) && isempty(conn.spike_affects) && isempty(conn.learning_rule)
end

Base.show(io::IO, c::Connector) = print(io, "$(c.source) => $(c.destination) with ", c.equation)

function show_field(io::IO, v::AbstractVector, title)
    if !isempty(v)
        println(io, title, " :")
        for val in v
            println(io, "\t $(val)")
        end
    end
end

function show_field(io::IO, d::Dict, title)
    if !isempty(d)
        println(io, title, " :")
        for (k, v) in d
            println(io, "\t ", k, " => ", v)
        end
    end
end

show_spike_affect(io::IO, t::Tuple) = println(io, "\t $(first(t)) += $(last(t))")

show_spike_affect(io::IO, eq::Equation) = println(io, "\t $eq")

function Base.show(io::IO, ::MIME"text/plain", c::Connector)
    
    println(io, "Connections :")
    for (s, d) in zip(c.source, c.destination)
        println(io, "\t $(s) => $(d)")
    end

    show_field(io, c.equation, "Equations")
    show_field(io, c.weight, "Weights")
    show_field(io, c.delay, "Delays")

    d = Dict()
    for w in c.weight  
        if haskey(c.learning_rule, w)
            d[w] = c.learning_rule[w]
        end
    end
    show_field(io, d, "Plasticity model")

    for s in c.source
        if haskey(c.spike_affects, s)
            println(io, "$(s) spikes affect :")
            sa = c.spike_affects[s]
            for x in sa
               show_spike_affect(io, x)
            end
        end
    end
end

function accumulate_equations!(eqs::AbstractVector{<:Equation}, bloxs)
    init_eqs = mapreduce(get_input_equations, vcat, bloxs)
    accumulate_equations!(eqs, init_eqs)

    return eqs
end

function accumulate_equations!(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs1, eq)
        else
            eqs1[idx] = eqs1[idx].lhs ~ eqs1[idx].rhs + eq.rhs
        end
    end

    return eqs1
end

function accumulate_equations(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    eqs = copy(eqs1)
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs, eq)
        else
            eqs[idx] = eqs[idx].lhs ~ eqs[idx].rhs + eq.rhs
        end
    end

    return eqs
end

ModelingToolkit.equations(c::Connector) = c.equation

discrete_callbacks(c::Connector) = c.discrete_callbacks

sources(c::Connector) = c.source

destinations(c::Connector) = c.destination

weights(c::Connector) = c.weight

delays(c::Connector) = c.delay

spike_affects(c::Connector) = c.spike_affects

learning_rules(c::Connector) = c.learning_rule

learning_rules(conns::AbstractVector{<:Connector}) = mapreduce(c -> c.learning_rule, merge!, conns)

get_equations_with_parameter_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> isparameter(eq.lhs), eqs)

get_equations_with_state_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> !isparameter(eq.lhs), eqs)

function get_states_spikes_affect(sa, name) 
    if haskey(sa, name)
        return first.(sa[name])
    else
        Num[]
    end
end

function get_params_spikes_affect(sa, name) 
    if haskey(sa, name)
        return last.(sa[name])
    else
        Num[]
    end
end

function generate_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    weight = get_weight(kwargs, name_out, name_in)
    w_name = Symbol("w_$(name_out)_$(name_in)")
    if typeof(weight) == Num   # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight [tunable=false])
    end    

    return w
end

function generate_gap_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    gap_weight = get_gap_weight(kwargs, name_out, name_in)
    gw_name = Symbol("g_w_$(name_out)_$(name_in)")
    if typeof(gap_weight) == Num   # Symbol
        gw = gap_weight
    else
        gw = only(@parameters $(gw_name)=gap_weight)
    end    

    return gw
end

"""
    Helper to merge delay and weight into a single vector
"""
function params(bc::Connector)
    wt = map(weights(bc)) do w
        Symbolics.get_variables(w)
    end

    if isempty(wt)
        return vcat(wt, delays(bc))
    else
        return vcat(reduce(vcat, wt), delays(bc))
    end
end

function Base.merge!(c1::Connector, c2::Connector)
    append!(c1.source, c2.source)
    append!(c1.destination, c2.destination)
    accumulate_equations!(c1.equation, c2.equation)
    append!(c1.weight, c2.weight)
    append!(c1.delay, c2.delay)
    append!(c1.discrete_callbacks, c2.discrete_callbacks)
    mergewith!(append!, c1.spike_affects, c2.spike_affects)
    merge!(c1.learning_rule, c2.learning_rule)
    return c1
end

Base.merge(c1::Connector, c2::Connector) = Base.merge!(deepcopy(c1), c2)

function hypergeometric_connections(neurons_src, neurons_dest, name_out, name_in; kwargs...)
    density = get_density(kwargs, name_out, name_in)
    N_connects =  density * length(neurons_dest) * length(neurons_src)
    out_degree = Int(ceil(N_connects / length(neurons_src)))
    in_degree =  Int(ceil(N_connects / length(neurons_dest)))
    wt = get_weight(kwargs,name_out, name_in)

    C = Connector[]
    outgoing_connections = zeros(Int, length(neurons_src))
    for neuron_postsyn in neurons_dest
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_src[idx]
                push!(C, Connector(neuron_presyn, neuron_postsyn; kwargs...))
            end
        else
            for i in idx 
                kwargs = (kwargs...,weight=wt[i])
                push!(C, Connector(neurons_src[i], neuron_postsyn; kwargs...))
            end
        end
        outgoing_connections[idx] .+= 1
    end

    return reduce(merge!, C)
end

function indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    rng = get(kwargs, :rng, Random.default_rng())
    in_degree =  Int(ceil(density * N_src))
    conn_mat = falses(N_src, N_dst)
    for j ∈ 1:N_dst
        idx = sample(rng, 1:N_src, in_degree; replace=false)
        for i ∈ idx
            conn_mat[i, j] = true
        end
    end
    conn_mat
end

function indegree_constrained_connections(neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    N_src = length(neurons_src)
    N_dst = length(neurons_dst)
    conn_mat = get(kwargs, :connection_matrix) do
        density = get_density(kwargs, name_src, name_dst)
        indegree_constrained_connection_matrix(density, N_src, N_dst; kwargs...)
    end

    C = Connector[]
    for j ∈ 1:N_dst
        for i ∈ 1:N_src
            if conn_mat[i, j]
                push!(C, Connector(neurons_src[i], neurons_dst[j]; kwargs...))
            end
        end
    end

    return reduce(merge!, C)
end

connection_rule(blox_src, blox_dest; kwargs...) = Connector(blox_src, blox_dest; kwargs...)

connection_equations(blox_src, blox_dest; kwargs...) = Connector(blox_src, blox_dest; kwargs...).equation

connection_equations(source, destination, w; kwargs...) = Equation[]

function connection_equations(blox_src::AbstractNeuronBlox, blox_dest::AbstractNeuronBlox, w; kwargs...)
    cr = get_connection_rule(kwargs, blox_src, blox_dest, w)

    return blox_dest.jcn ~ cr
end

connection_spike_affects(source, destination, w) = Tuple{Num, Num}[]

function connection_learning_rule(source, destination, w; kwargs...)
    if haskey(kwargs, :learning_rule)
        return Dict(w => deepcopy(kwargs[:learning_rule]))
    else
        return Dict{Num, AbstractLearningRule}()
    end
end
    
connection_callbacks(source, destination; kwargs...) = []

function Connector(blox_src::AbstractBlox, blox_dest::AbstractBlox; kwargs...)
    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = connection_equations(blox_src, blox_dest, w; kwargs...)
    lr = connection_learning_rule(blox_src, blox_dest, w; kwargs...)  
    cb = connection_callbacks(blox_src, blox_dest; kwargs...)

    affects_tuple = connection_spike_affects(blox_src, blox_dest, w)
    sa = Dict(namespaced_nameof(blox_src) => to_vector(affects_tuple))  
    
    return Connector(
        namespaced_nameof(blox_src), 
        namespaced_nameof(blox_dest);
        equation = eq, 
        weight = w,
        spike_affects = sa,
        discrete_callbacks = cb,
        learning_rule = lr
    )
end

function Connector(
    blox_src::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    blox_dest::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}; 
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
    blox_src::Union{HHNeuronInhib_MSN_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    blox_dest::Union{HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
        
    STA = get_sta(kwargs, nameof(blox_src), nameof(blox_dest))
    eq = if STA
        sys_dest.I_syn ~ -w * sys_dest.Gₛₜₚ * sys_src.G * (sys_dest.V - sys_src.E_syn)
    else
        sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)
    end

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::HHNeuronInhib_FSI_Adam_Blox,
    blox_dest::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    eq = sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn)
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::HHNeuronInhib_FSI_Adam_Blox,
    blox_dest::HHNeuronInhib_FSI_Adam_Blox; 
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
    blox_dest::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; 
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

sigmoid(x, r) = one(x) / (one(x) + exp(-r*x))

function Connector(
    blox_src::JansenRitSPM12, 
    blox_dest::JansenRitSPM12; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    x = only(outputs(blox_src; namespaced=true))
    r = namespace_expr(blox_src.params[2], sys_src)

    eq = sys_dest.jcn ~ sigmoid(x, r)*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=[w, r])
end

function Connector(
    blox_src::NeuralMassBlox, 
    blox_dest::NeuralMassBlox; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    lr = get_learning_rule(kwargs, nameof(sys_src), nameof(sys_dest))
    x = only(outputs(blox_src; namespaced=true))
    if x isa Num
        eq = sys_dest.jcn ~ x*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, learning_rule=Dict(w => lr))
    else
        @variables t
        delay = get_delay(kwargs, nameof(sys_src), nameof(sys_dest))
        τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        τ = only(@parameters $(τ_name)=delay)

        eq = sys_dest.jcn ~ x(t-τ)*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ, learning_rule=Dict(w => lr))
    end    
end

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

# additional dispatch to connect to hemodynamic observer blox
function Connector(
    blox_src::NeuralMassBlox, 
    blox_dest::ObserverBlox;
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    x = only(outputs(blox_src; namespaced=true))
    if x isa Num
        w = generate_weight_param(blox_src, blox_dest; kwargs...)
        eq = sys_dest.jcn ~ x*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    else
        # Need t for the delay term
        @variables t
        # Define & accumulate delay parameter
        # Don't accumulate if zero
        τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        τ = only(@parameters $(τ_name)=delay)

        w_name = Symbol("w_$(nameof(sys_src))_$(nameof(sys_dest))")
        w = only(@parameters $(w_name)=weight)
        
        eq = sys_dest.jcn ~ x(t-τ)*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ)
    end
end

# additional dispatch to connect to a stimulus blox, first crafted for ExternalInput
function Connector(
    blox_src::StimulusBlox,
    blox_dest::NeuralMassBlox;
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    blox_dest::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)
    neurons_src = get_inh_neurons(blox_src)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::STN_Adam,
    blox_dest::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_src = get_exci_neurons(blox_src)
    neurons_dest = get_inh_neurons(blox_dest)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    blox_dest::STN_Adam;
    kwargs...
)
    neurons_src = get_inh_neurons(blox_src)
    neurons_dest = get_exci_neurons(blox_dest)

    conn = indegree_constrained_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::NGNMM_theta, 
    blox_dest::CorticalBlox; 
    kwargs...
)
    neurons_dest = get_inh_neurons(blox_dest)

    conn = Connector(blox_src, neurons_dest[end]; kwargs...)

    return conn
end

function Connector(
    blox_src::CanonicalMicroCircuitBlox,
    blox_dest::CanonicalMicroCircuitBlox;
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
    blox_src::StimulusBlox,
    blox_dest::CanonicalMicroCircuitBlox;
    kwargs...
)
    sysparts_dest = get_parts(blox_dest)
    conn = Connector(blox_src, sysparts_dest[1]; kwargs...)

    return conn
end

function Connector(
    blox_src::CanonicalMicroCircuitBlox,
    blox_dest::ObserverBlox;
    kwargs...
)
    sysparts_src = get_parts(blox_src)
    conn = Connector(sysparts_src[2], blox_dest; kwargs...)

    return conn
end

function Connector(
    blox_src::WinnerTakeAllBlox, 
    blox_dest::WinnerTakeAllBlox; 
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
    blox_src::HHNeuronInhibBlox, 
    blox_dest::WinnerTakeAllBlox; 
    kwargs...
)
    neurons_dest = get_exci_neurons(blox_dest)

    conn = mapreduce(merge!, neurons_dest) do neuron_postsyn
        Connector(blox_src, neuron_postsyn; kwargs...)
    end

    return conn
end

function Connector(
    blox_src::Union{CorticalBlox,STN,Thalamus},
    blox_dest::Union{CorticalBlox,STN,Thalamus};
    kwargs...
)
    neurons_dest = get_exci_neurons(blox_dest)
    neurons_src = get_exci_neurons(blox_src)

    conn = hypergeometric_connections(neurons_src, neurons_dest, nameof(blox_src), nameof(blox_dest); kwargs...)

    return conn
end

function Connector(
    blox_src::Union{CorticalBlox,STN,Thalamus},
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
    blox_dest::Union{CorticalBlox,STN,Thalamus};
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
    cb::CorticalBlox,
    str::Striatum;
    kwargs...
)
    neurons_dest = get_inh_neurons(str)
    neurons_src = get_exci_neurons(cb)

    w = get_weight(kwargs, namespaced_nameof(cb), namespaced_nameof(str))

    dist = Uniform(0,1)
    wt_ar = 2*w*rand(dist, length(neurons_src)) # generate a uniform distribution of weight with average value w 
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
    neuron::HHNeuronExciBlox,
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
    blox_src::HHNeuronExciBlox,
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
    v = rand(Poisson(R))
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
    cb = [t_event+sqrt(eps(t_event))] => (sample_affect!, [], [sys_src.κ, sys_src.jcn, sys_dest.TAN_spikes], [])

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
    neuron::Union{HHNeuronExciBlox, HHNeuronInhibBlox};
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
    cb::CorticalBlox;
    kwargs...
)
    neurons = get_exci_neurons(cb)

    conn = mapreduce(merge!, neurons) do neuron
        Connector(stim, neuron; kwargs...)
    end

    return conn
end

Connector(blox::AbstractBlox, as::AbstractActionSelection; kwargs...) = Connector(namespaced_nameof(blox), namespaced_nameof(as))

# Connects a neural mass as a driving input to a spiking neuron
# Should be used with care because units will be strange (NMM typically outputs voltage but neuron inputs are typically currents)
function Connector(
    blox_src::AbstractNeuronBlox, 
    blox_dest::NeuralMassBlox; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(outputs(blox_src; namespaced=true))
    if x isa Num
        eq = sys_dest.jcn ~ x*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
    else
        @variables t
        delay = get_delay(kwargs, nameof(blox_src), nameof(blox_dest))
        τ_name = Symbol("τ_$(nameof(sys_src))_$(nameof(sys_dest))")
        τ = only(@parameters $(τ_name)=delay)

        eq = sys_dest.jcn ~ x(t-τ)*w

        return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w, delay=τ)
    end
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
    blox_src::Union{LIFExciCircuitBlox, LIFInhCircuitBlox}, 
    blox_dest::Union{LIFExciCircuitBlox, LIFInhCircuitBlox};
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
    blox_dest::Union{LIFExciCircuitBlox, LIFInhCircuitBlox};
    kwargs...
)
    neurons_dest = get_neurons(blox_dest)

    conn = mapreduce(merge!, neurons_dest) do neuron
        Connector(stim, neuron; kwargs...)
    end

    return conn
end

# New version - need to discuss gₛ implementation
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
    blox_src::DBS,
    blox_dest::CompositeBlox;
    kwargs...
)
    components = get_components(blox_dest)
    conn = mapreduce(merge!, components) do comp
        Connector(blox_src, comp; kwargs...)
    end

    return conn
end

function Connector(
    blox_src::DBS,
    blox_dest::AbstractNeuronBlox;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    
    eq = sys_dest.I_in ~ w * sys_src.u
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::DBS,
    blox_dest::NeuralMassBlox;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    
    eq = sys_dest.jcn ~ w * sys_src.u

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

function Connector(
    blox_src::DBS,
    blox_dest::HHNeuronExci_STN_Adam_Blox;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    
    eq = sys_dest.DBS_in ~ - sys_dest.V/sys_dest.b + sys_src.u
    
    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq)
end

# Create excitatory -> inhibitory AMPA receptor conenction
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

# Create inhibitory -> inhibitory/excitatory GABA_A receptor connection
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
    blox_src::MetabolicHHNeuron,
    blox_dest::MetabolicHHNeuron;
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    
    eq = sys_dest.I_syn ~ -w * sys_src.G * (sys_dest.V - sys_src.E_syn) * sys_src.S * exp(-sys_src.χ/5)

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end
