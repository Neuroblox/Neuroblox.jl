struct HHConnection_STA{T} <: ConnectionRule
    weight::T
    HHConnection_STA{T}(x) where {T} = new{T}(x)
    HHConnection_STA(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::HHConnection_STA, name_src, name_dst)
    (; weight = Symbol(:w_STA_, name_src, :_, name_dst))
end
Base.zero(::Type{HHConnection_STA{T}}) where {T} = HHConnection_STA(zero(T))
Base.zero(::Type{HHConnection_STA}) = HHConnection_STA(0.0)

function GraphDynamics.system_wiring_rule!(g, 
    HH_src::Union{HHNeuronExci, HHNeuronInhib, HHNeuronFSI}, 
    HH_dst::Union{HHNeuronExci, HHNeuronInhib, HHNeuronFSI};
    weight, sta=false, learning_rule=NoLearningRule(), kwargs...)
    
    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(HH_src.name, "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(HH_dst.name, "spikes_cumulative"))
    if sta & !(HH_src isa HHNeuronFSI) # Don't hit STA rules for FSI
        conn = HHConnection_STA(weight)
    else
        conn = BasicConnection(weight)
    end
    add_connection!(g, HH_src, HH_dst; conn, weight, learning_rule, kwargs...)
end

function (c::BasicConnection)(HH_src::Union{Subsystem{HHNeuronExci},
                                            Subsystem{HHNeuronInhib},
                                            Subsystem{HHNeuronFSI}}, 
                              HH_dst::Union{Subsystem{HHNeuronExci},
                                            Subsystem{HHNeuronInhib},
                                            Subsystem{HHNeuronFSI}},
                              t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.G * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function (c::HHConnection_STA)(HH_src::Union{Subsystem{HHNeuronExci},
                                              Subsystem{HHNeuronInhib}}, 
                                HH_dst::Union{Subsystem{HHNeuronExci},
                                              Subsystem{HHNeuronInhib}},
                               t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_dst.Gₛₜₚ * HH_src.G * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function (c::BasicConnection)(HH_src::Subsystem{HHNeuronFSI},
                              HH_dst::Subsystem{HHNeuronFSI}, t)
    acc = initialize_input(HH_dst)
    I_syn = -c.weight * HH_src.Gₛ * (HH_dst.V - HH_src.E_syn)
    @set acc.I_syn = I_syn
end

function GraphDynamics.system_wiring_rule!(g,
                                           HH_src::HHNeuronFSI, 
                                           HH_dst::HHNeuronFSI; weight, gap=false, kwargs...)

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

function ((;w_gap)::HHConnection_GAP)(HH_src::Subsystem{HHNeuronFSI}, 
                                      HH_dst::Subsystem{HHNeuronFSI}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap * (HH_dst.V - HH_src.V)
    acc
end

function ((;w_gap_rev)::HHConnection_GAP_Reverse)(HH_src::Subsystem{HHNeuronFSI}, 
                                                  HH_dst::Subsystem{HHNeuronFSI}, t)
    acc = initialize_input(HH_dst)
    acc = @set acc.I_gap = -w_gap_rev * (HH_dst.V - HH_src.V)
    acc
end

##----------------------------------------------
# Next Generation EI
function (c::BasicConnection)((;aₑ, bₑ, Cₑ)::Subsystem{NGNMM_theta}, 
                              HH_dst::Union{Subsystem{HHNeuronExci}, Subsystem{HHNeuronInhib}, Subsystem{HHNeuronFSI}}, t)
    w = c.weight
    acc = initialize_input(HH_dst)
    a = aₑ
    b = bₑ
    C = Cₑ
    f = (1/(C*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2) 
    @set acc.I_asc = w*f
end

##----------------------------------------------
# WinnerTakeAll

function GraphDynamics.system_wiring_rule!(g, wta::WinnerTakeAll; kwargs...)
    inh = wta.parts[1]
    for exci ∈ wta.parts[2:end]
        system_wiring_rule!(g, inh, exci; weight = 1.0)
        system_wiring_rule!(g, exci, inh; weight = 1.0)
    end
end

function GraphDynamics.system_wiring_rule!(g, wta_src::WinnerTakeAll, wta_dst::WinnerTakeAll; kwargs...)
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

function GraphDynamics.system_wiring_rule!(g, neuron_src::HHNeuronInhib, wta_dst::WinnerTakeAll; kwargs...)
    neurons_dst = get_exci_neurons(wta_dst)
    for neuron_dst ∈ neurons_dst
        system_wiring_rule!(g, neuron_src, neuron_dst; kwargs...)
    end
end

##----------------------------------------------
# Cortical
function GraphDynamics.system_wiring_rule!(g, c::Cortical; kwargs...)
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
                                           blox_src::Union{Cortical,STN,Thalamus},
                                           blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    name_dst = namespaced_nameof(blox_dst)
    name_src = namespaced_nameof(blox_src)

    cr = get_connection_rule(kwargs, blox_src, blox_dst)

    if cr == :density
        conn = density_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    elseif cr == :weightmatrix
        conn = weight_matrix_connections!(g, neurons_src, neurons_dst, nameof(blox_src), nameof(blox_dst); kwargs...)
    else
        conn = hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    end
end


function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{Cortical,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_inh_neurons(blox_src)
    name_dst = namespaced_nameof(blox_dst)
    name_src = namespaced_nameof(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
end

function GraphDynamics.system_wiring_rule!(g, blox_src::NGNMM_theta, blox_dst::Cortical; kwargs...)
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

function GraphDynamics.system_wiring_rule!(g, blox::Thalamus; kwargs...)
    connection_matrix = blox.connection_matrix
    excis = get_exci_neurons(blox)
    for i ∈ eachindex(excis)
        system_wiring_rule!(g, excis[i])
        for j ∈ eachindex(excis)
            cij = connection_matrix[i,j]
            if !iszero(cij)
                system_wiring_rule!(g, excis[i], excis[j], weight=cij)
            end
        end
    end
end

function (c::BasicConnection)(src::Subsystem{<:AbstractSimpleStimulus}, dst::Subsystem{<:Union{HHNeuronInhib, HHNeuronExci}}, t)
    w = c.weight
    input = initialize_input(dst)
    x = only(outputs(src))
    @reset input.I_in = w * x
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, src::AbstractStimulus, dst::Thalamus; kwargs...) 
    for neuron_dst ∈ get_exci_neurons(dst)
        system_wiring_rule!(g, src, neuron_dst; kwargs...)
    end
end

#----------------------------------------------
# Discrete blox

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
                                           sys_src::HHNeuronExci,
                                           sys_dst::Union{Matrisome, Striosome};
                                           weight,
                                           learning_rule=NoLearningRule(),
                                           kwargs...)
    
    conn = BasicConnection(weight)
     learning_rule = maybe_set_state_pre( learning_rule, Symbol(namespaced_nameof(sys_src), :₊spikes_cumulative))
     learning_rule = maybe_set_state_post(learning_rule, Symbol(namespaced_nameof(sys_dst), :₊H_learning))
    add_connection!(g, sys_src, sys_dst; weight, conn, learning_rule, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{HHNeuronExci}, sys_dst::Subsystem{<:Union{Matrisome, Striosome}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.spikes_window
end


function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::Matrisome, sys_dst::Union{Matrisome, Striosome, HHNeuronInhib}; weight=1.0, kwargs...)
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    conn = EventConnection(weight, (;t_init=0.1, t_event))
    add_connection!(g, sys_src, sys_dst; conn, kwargs...)
end

function (c::EventConnection)(src::Subsystem{Matrisome}, dst::Subsystem{<:Union{Matrisome, Striosome, HHNeuronInhib}}, t)
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
                                             m_dst::Subsystem{HHNeuronInhib})
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

function GraphDynamics.system_wiring_rule!(g::GraphSystem, sys_src::TAN, sys_dst::Matrisome; weight=1.0, kwargs...)
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

function GraphDynamics.system_wiring_rule!(g::GraphSystem, stim::ImageStimulus, neuron::Union{HHNeuronInhib, HHNeuronExci}; current_pixel, weight, kwargs...)
    add_connection!(g, stim, neuron; conn=StimConnection(weight, current_pixel), weight, kwargs...)
end

struct StimConnection <: ConnectionRule
    weight::Float64
    pixel_index::Int
end

function (c::StimConnection)(src::Subsystem{ImageStimulus},
                             dst::Subsystem{<:Union{HHNeuronExci, HHNeuronInhib}},
                             t)
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

#----------------------------------------------
# Striatum|GPi|GPe - Cortical|STN|Thalamus

function GraphDynamics.system_wiring_rule!(g, cb_src::Union{Cortical,STN,Thalamus}, cb_dst::Union{GPi, GPe}; kwargs...)
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

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::HHNeuronExci, blox_dst::Union{Striatum, GPi}; kwargs...)
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

function density_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_dst = length(neurons_dst)
    rng = get(kwargs, :rng, default_rng())

    for ns in neurons_src
        idxs = findall(rand(rng, N_dst) .<= density)
        for i in idxs
            system_wiring_rule!(g, ns, neurons_dst[i]; kwargs...)
        end
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, cb::Cortical, str::Striatum; kwargs...)
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
    t_event = get_event_time(kwargs, nameof(sys_src), nameof(sys_dst))
    
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


