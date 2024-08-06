mutable struct BloxConnector
    eqs::Vector{Equation}
    weights::Vector{Num}
    delays::Vector{Num}
    discrete_callbacks
    continuous_callbacks
    spike_affect_states::Dict{Symbol, Vector{Num}}
    learning_rules

    BloxConnector() = new(Equation[], Num[], Num[], Pair{Any, Vector{Equation}}[], Dict{Symbol, Vector{Num}}(), Dict{Num, AbstractLearningRule}())

    function BloxConnector(bloxs)
        eqs = mapreduce(get_input_equations, vcat, bloxs) 
        weights = mapreduce(get_weight_parameters, vcat, bloxs)
        delays = mapreduce(get_delay_parameters, vcat, bloxs)
        discrete_callbacks = mapreduce(get_discrete_callbacks, vcat, bloxs)
        continuous_callbacks = mapreduce(get_continuous_callbacks, vcat, bloxs)
        spike_affect_states = mapreduce(get_spike_affect_states, merge, bloxs)
        learning_rules = mapreduce(get_weight_learning_rules, merge, bloxs)

        new(eqs, weights, delays, discrete_callbacks, continuous_callbacks, spike_affect_states, learning_rules)
    end
end

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs + eq.rhs
end

function accumulate_spike_affect_states!(bc::BloxConnector, name_blox_src, states_dst)
    if haskey(bc.spike_affect_states, name_blox_src)
        append!(bc.spike_affect_states[name_blox_src], states_dst)
    else
        bc.spike_affect_states[name_blox_src] = states_dst
    end
end

get_equations_with_parameter_lhs(bc) = filter(eq -> isparameter(eq.lhs), bc.eqs)

get_equations_with_state_lhs(bc) = filter(eq -> !isparameter(eq.lhs), bc.eqs)

function generate_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    weight = get_weight(kwargs, name_out, name_in)
    w_name = Symbol("w_$(name_out)_$(name_in)")
    if typeof(weight) == Num   # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight)
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

function hypergeometric_connections!(bc, neurons_out, neurons_in, name_out, name_in; kwargs...)
    density = get_density(kwargs, name_out, name_in)
    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))
    wt = get_weight(kwargs,name_out, name_in)

    outgoing_connections = zeros(Int, length(neurons_out))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_out[idx]
                bc(neuron_presyn, neuron_postsyn; kwargs...)
            end
        else
            for i in idx 
                kwargs = (kwargs...,weight=wt[i])
                bc(neurons_out[i], neuron_postsyn; kwargs...)
            end
        end
        outgoing_connections[idx] .+= 1
    end
end

function indegree_constrained_connections!(bc, neurons_out, neurons_in, name_out, name_in; kwargs...)
    density = get_density(kwargs, name_out, name_in)
    in_degree =  Int(ceil(density * length(neurons_out)))
    for neuron_postsyn in neurons_in
        idx = sample(collect(1:length(neurons_out)), in_degree; replace=false)
        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; kwargs...)
        end
    end
end

"""
    Helper to merge delays and weights into a single vector
"""
function params(bc::BloxConnector)
    weights = []
    for w in bc.weights
        append!(weights, Symbolics.get_variables(w))
    end
    if isempty(weights)
        return vcat(weights, bc.delays)
    else
        return vcat(reduce(vcat, weights), bc.delays)
    end
end

function (bc::BloxConnector)(
    HH_out::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}, 
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronInhib_FSI_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}; 
    kwargs...
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    w = generate_weight_param(HH_out, HH_in; kwargs...)
    push!(bc.weights, w)    

    if haskey(kwargs, :learning_rule)
        lr = deepcopy(kwargs[:learning_rule])
        maybe_set_state_pre!(lr, sys_out.spikes_cumulative)
        maybe_set_state_post!(lr,sys_in.spikes_cumulative)
        bc.learning_rules[w] = lr
    end

    STA = get_sta(kwargs, nameof(HH_out), nameof(HH_in))

    
    eq = if STA
        sys_in.I_syn ~ -w * sys_in.Gₛₜₚ * sys_out.G * (sys_in.V - sys_out.E_syn)
    else
        sys_in.I_syn ~ -w * sys_out.G * (sys_in.V - sys_out.E_syn)
    end

    accumulate_equation!(bc, eq)
    
    GAP = get_gap(kwargs, nameof(HH_out), nameof(HH_in))
    if GAP
        w_gap = generate_gap_weight_param(HH_out, HH_in; kwargs...)
        push!(bc.weights, w_gap)
        eq2 = sys_in.I_gap ~ -w_gap * (sys_in.V - sys_out.V)
        accumulate_equation!(bc, eq2) 
        eq3 = sys_out.I_gap ~ -w_gap * (sys_out.V - sys_in.V)
        accumulate_equation!(bc, eq3) 
    end

end

function (bc::BloxConnector)(
    HH_out::HHNeuronInhib_FSI_Adam_Blox,
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox, HHNeuronInhib_MSN_Adam_Blox, HHNeuronExci_STN_Adam_Blox, HHNeuronInhib_GPe_Adam_Blox}; 
    kwargs...
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    w = generate_weight_param(HH_out, HH_in; kwargs...)
    push!(bc.weights, w)    

    eq = sys_in.I_syn ~ -w * sys_out.G * (sys_in.V - sys_out.E_syn)
    
    accumulate_equation!(bc, eq)

    GAP = get_gap(kwargs, nameof(HH_out), nameof(HH_in))
    if GAP
        w_gap = generate_gap_weight_param(HH_out, HH_in; kwargs...)
        push!(bc.weights, w_gap)
        eq2 = sys_in.I_gap ~ -w_gap * (sys_in.V - sys_out.V)
        accumulate_equation!(bc, eq2) 
        eq3 = sys_out.I_gap ~ -w_gap * (sys_out.V - sys_in.V)
        accumulate_equation!(bc, eq3) 
    end
end

function (bc::BloxConnector)(
    HH_out::HHNeuronInhib_FSI_Adam_Blox,
    HH_in::HHNeuronInhib_FSI_Adam_Blox; 
    kwargs...
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    w = generate_weight_param(HH_out, HH_in; kwargs...)
    push!(bc.weights, w)    

        
    eq = sys_in.I_syn ~ -w * sys_out.Gₛ * (sys_in.V - sys_out.E_syn)
    
    accumulate_equation!(bc, eq)

    GAP = get_gap(kwargs, nameof(HH_out), nameof(HH_in))
    if GAP
        w_gap = generate_gap_weight_param(HH_out, HH_in; kwargs...)
        push!(bc.weights, w_gap)
        eq2 = sys_in.I_gap ~ -w_gap * (sys_in.V - sys_out.V)
        accumulate_equation!(bc, eq2) 
        eq3 = sys_out.I_gap ~ -w_gap * (sys_out.V - sys_in.V)
        accumulate_equation!(bc, eq3) 
    end
end

function (bc::BloxConnector)(
    cb_out::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    cb_in::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    indegree_constrained_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb_out::STN_Adam,
    cb_in::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)

    indegree_constrained_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb_out::Union{Striatum_MSN_Adam,Striatum_FSI_Adam,GPe_Adam},
    cb_in::STN_Adam;
    kwargs...
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    indegree_constrained_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    asc_out::NextGenerationEIBlox, 
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; 
    kwargs...
)
    sys_out = get_namespaced_sys(asc_out)
    sys_in = get_namespaced_sys(HH_in)

    w = generate_weight_param(asc_out, HH_in; kwargs...)
    push!(bc.weights, w)  

    #Z = sys_out.Z 
    a = sys_out.aₑ
    b = sys_out.bₑ
    f = (1/(sys_out.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)   
    eq = sys_in.I_asc ~ w*f
        
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    asc_out::NextGenerationEIBlox, 
    cb_in::CorticalBlox; 
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)

    bc(asc_out, neurons_in[end]; kwargs...)
end

function (bc::BloxConnector)(
    bloxout::CanonicalMicroCircuitBlox,
    bloxin::CanonicalMicroCircuitBlox;
    kwargs...
)
    sysparts_out = get_blox_parts(bloxout)
    sysparts_in = get_blox_parts(bloxin)

    wm = get_weightmatrix(kwargs, namespaced_nameof(bloxin), namespaced_nameof(bloxout))

    idxs = findall(!iszero, wm)
    for idx in idxs
        bc(sysparts_out[idx[2]], sysparts_in[idx[1]]; weight=wm[idx])
    end
end

# define a sigmoid function
sigmoid(x, r) = one(x) / (one(x) + exp(-r*x))

function (bc::BloxConnector)(
    bloxout::JansenRitSPM12, 
    bloxin::JansenRitSPM12; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    x = namespace_expr(bloxout.output, sys_out)
    r = namespace_expr(bloxout.params[2], sys_out)
    push!(bc.weights, r)

    eq = sys_in.jcn ~ sigmoid(x, r)*w
    
    accumulate_equation!(bc, eq)
end


function (bc::BloxConnector)(
    bloxout::NeuralMassBlox, 
    bloxin::NeuralMassBlox; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        lr = deepcopy(kwargs[:learning_rule])
        bc.learning_rules[w] = lr
    end

    if typeof(bloxout.output) == Num
        x = namespace_expr(bloxout.output, sys_out)
        eq = sys_in.jcn ~ x*w
    else
        @variables t
        delay = get_delay(kwargs, nameof(bloxout), nameof(bloxin))
        τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
        τ = only(@parameters $(τ_name)=delay)
        push!(bc.delays, τ)

        x = namespace_expr(bloxout.output, sys_out)
        eq = sys_in.jcn ~ x(t-τ)*w
    end
    
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    bloxout::KuramotoOscillator, 
    bloxin::KuramotoOscillator; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    xₒ = namespace_expr(bloxout.output, sys_out)
    xᵢ = namespace_expr(bloxin.output, sys_in) #needed because this is also the θ term of the block receiving the connection

    eq = sys_in.jcn ~ w*sin(xₒ - xᵢ)
    accumulate_equation!(bc, eq)
end

# additional dispatch to connect to hemodynamic observer blox
function (bc::BloxConnector)(
    bloxout::NeuralMassBlox, 
    bloxin::ObserverBlox; 
    weight=1,
    delay=0,
    density=0.1
)
    # Need t for the delay term
    @variables t

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    if typeof(bloxout.output) == Num
        w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
        if typeof(weight) == Num # Symbol
            w = weight
        else
            w = only(@parameters $(w_name)=weight [tunable=false])
        end    
        push!(bc.weights, w)
        x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
        eq = sys_in.jcn ~ x*w
    else
        # Define & accumulate delay parameter
        # Don't accumulate if zero
        τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
        τ = only(@parameters $(τ_name)=delay)
        push!(bc.delays, τ)

        w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
        w = only(@parameters $(w_name)=weight)
        push!(bc.weights, w)

        x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
        eq = sys_in.jcn ~ x(t-τ)*w
    end
    
    accumulate_equation!(bc, eq)
end

# additional dispatch to connect to a stimulus blox, first crafted for ExternalInput
function (bc::BloxConnector)(
    bloxout::StimulusBlox,
    bloxin::NeuralMassBlox;
    weight=1
)

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    if typeof(weight) == Num # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight)
    end    
    push!(bc.weights, w)

    x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
    eq = sys_in.jcn ~ x*w

    accumulate_equation!(bc, eq)
end

# # Ok yes this is a bad dispatch but the whole compound blocks implementation is hacky and needs fixing @@
# # Opening an issue to loop back to this during clean up week
# function (bc::BloxConnector)(
#     bloxout::CompoundNOBlox, 
#     bloxin::CompoundNOBlox; 
#     weight=1,
#     delay=0,
#     density=0.1
# )

#     sys_out = get_namespaced_sys(bloxout)
#     sys_in = get_namespaced_sys(bloxin)

#     w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
#     if typeof(weight) == Num # Symbol
#         w = weight
#     else
#         w = only(@parameters $(w_name)=weight)
#     end
#     push!(bc.weights, w)
#     x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
#     eq = sys_in.nmm₊jcn ~ x*w
    
#     accumulate_equation!(bc, eq)
# end

function (bc::BloxConnector)(
    wta_out::WinnerTakeAllBlox, 
    wta_in::WinnerTakeAllBlox; 
    kwargs...
)
    neurons_in = get_exci_neurons(wta_in)
    neurons_out = get_exci_neurons(wta_out)

    density = get_density(kwargs, nameof(wta_out), nameof(wta_in))
    dist = Bernoulli(density)

    for neuron_postsyn in neurons_in
        name_postsyn = namespaced_nameof(neuron_postsyn)
        for neuron_presyn in neurons_out
            name_presyn = namespaced_nameof(neuron_presyn)
            # Check names to avoid recurrent connections between the same neuron
            if (name_postsyn != name_presyn) && rand(dist)
                bc(neuron_presyn, neuron_postsyn; kwargs...)
            end
        end
    end
end

function (bc::BloxConnector)(
    neuron_out::HHNeuronInhibBlox, 
    wta_in::WinnerTakeAllBlox; 
    kwargs...
)
    neurons_in = get_exci_neurons(wta_in)

    for neuron_postsyn in neurons_in
        bc(neuron_out, neuron_postsyn; kwargs...)
    end
end

function (bc::BloxConnector)(
    cb_out::Union{CorticalBlox,STN,Thalamus},
    cb_in::Union{CorticalBlox,STN,Thalamus};
    kwargs...
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb_out::Union{CorticalBlox,STN,Thalamus},
    cb_in::Union{GPi, GPe};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb_out::Union{Striatum, GPi, GPe},
    cb_in::Union{CorticalBlox,STN,Thalamus};
    kwargs...
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb_out::Union{Striatum, GPi, GPe},
    cb_in::Union{GPi, GPe};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb_out), nameof(cb_in); kwargs...)
end

function (bc::BloxConnector)(
    cb::CorticalBlox,
    str::Striatum;
    kwargs...
)
    neurons_in = get_inh_neurons(str)
    neurons_out = get_exci_neurons(cb)

    w = get_weight(kwargs, namespaced_nameof(cb), namespaced_nameof(str))

    dist = Uniform(0,1)
    wt_ar = 2*w*rand(dist, length(neurons_out)) # generate a uniform distribution of weights with average value w 
    kwargs = (kwargs..., weight=wt_ar)

    if haskey(kwargs, :learning_rule)
        lr = deepcopy(kwargs[:learning_rule])
        sys_matr = get_namespaced_sys(get_matrisome(str))
        maybe_set_state_post!(lr, sys_matr.H_learning)
        kwargs = (kwargs..., learning_rule=lr)
    end

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb), nameof(str); kwargs...)

    algebraic_parts = [get_matrisome(str), get_striosome(str)]

    for (i,neuron_presyn) in enumerate(neurons_out)
        kwargs = (kwargs...,weight=wt_ar[i])
        for part in algebraic_parts
            bc(neuron_presyn, part; kwargs...)
        end
    end
end

function (bc::BloxConnector)(
    neuron::HHNeuronExciBlox,
    str::Striatum;
    kwargs...
)
    neurons_in = get_inh_neurons(str)
    neuron_out = neuron

    for neuron_postsyn in neurons_in
        bc(neuron_out, neuron_postsyn; kwargs...)
    end
       
end

function (bc::BloxConnector)(
    neuron::HHNeuronExciBlox,
    gpi::GPi;
    kwargs...
)
    neurons_in = get_inh_neurons(gpi)
    neuron_out = neuron

    for neuron_postsyn in neurons_in
        bc(neuron_out, neuron_postsyn; kwargs...)
    end
       
end

function (bc::BloxConnector)(
    neuron::HHNeuronExciBlox,
    discr::Union{Matrisome, Striosome};
    kwargs...
)
    sys_out = get_namespaced_sys(neuron)
    sys_in = get_namespaced_sys(discr)

    w = generate_weight_param(neuron, discr; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        lr = deepcopy(kwargs[:learning_rule])
        maybe_set_state_pre!(lr, sys_out.spikes_cumulative)
        maybe_set_state_post!(lr, sys_in.H_learning)
        bc.learning_rules[w] = lr
    end

    eq = sys_in.jcn ~ w*sys_out.spikes_window
    accumulate_equation!(bc, eq)    
end

function (bc::BloxConnector)(
    str_out::Striatum,
    str_in::Striatum;
    kwargs... 
)
    sys_matr_out = get_namespaced_sys(get_matrisome(str_out))
    sys_matr_in = get_namespaced_sys(get_matrisome(str_in))
    sys_strios_in = get_namespaced_sys(get_striosome(str_in))
    neurons_in = get_inh_neurons(str_in)

    t_event = get_event_time(kwargs, nameof(str_out), nameof(str_in))
    cb_matr = [t_event] => [sys_matr_in.H ~ ifelse(sys_matr_out.H*sys_matr_out.jcn > sys_matr_in.H*sys_matr_in.jcn, 0, 1)]
    cb_strios = [t_event] => [sys_strios_in.H ~ ifelse(sys_matr_out.H*sys_matr_out.jcn > sys_matr_in.H*sys_matr_in.jcn, 0, 1)]
    
    # HACK: H should be reset to 1 at the beginning of each trial
    # Such callbacks should be moved to RL-specific functions like `run_experiment!`
    cb_matr_init = [0.1] => [sys_matr_in.H ~ 1]
    cb_strios_init = [0.1] => [sys_strios_in.H ~ 1]

    push!(bc.discrete_callbacks, cb_matr)
    push!(bc.discrete_callbacks, cb_strios)
    push!(bc.discrete_callbacks, cb_matr_init)
    push!(bc.discrete_callbacks, cb_strios_init)

    for neuron in neurons_in
        sys_neuron = get_namespaced_sys(neuron)
        # Large negative current added to shut down the Striatum spiking neurons.
        # Value is hardcoded for now, as it's more of a hack, not user option. 
        cb_neuron = [t_event] => [sys_neuron.I_bg ~ ifelse(sys_matr_out.H*sys_matr_out.jcn > sys_matr_in.H*sys_matr_in.jcn, -2, 0)]
        # lateral inhibition current I_bg should be set to 0 at the beginning of each trial
        cb_neuron_init = [0.1] => [sys_neuron.I_bg ~ 0]
        push!(bc.discrete_callbacks, cb_neuron)
        push!(bc.discrete_callbacks, cb_neuron_init)
    end
end

function (bc::BloxConnector)(
    str::Striatum,
    discr::Union{TAN, SNc};
    kwargs... 
)
    striosome = get_striosome(str)
    bc(striosome, discr; kwargs...)
end

function (bc::BloxConnector)(
    discr_out::Striosome,
    discr_in::Union{TAN, SNc};
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    w = generate_weight_param(discr_out, discr_in; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        bc.learning_rules[w] = deepcopy(kwargs[:learning_rule])
    end

    eq = sys_in.jcn ~ w*sys_out.H*sys_out.jcn
  

    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    tan::TAN,
    str::Striatum;
    kwargs...
) 
    matrisome = get_matrisome(str)
    bc(tan, matrisome; kwargs...)
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

function (bc::BloxConnector)(
    discr_out::TAN,
    discr_in::Matrisome;
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    w = generate_weight_param(discr_out, discr_in; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        bc.learning_rules[w] = deepcopy(kwargs[:learning_rule])
    end

    t_event = get_event_time(kwargs, nameof(discr_out), nameof(discr_in))
    cb = [t_event+sqrt(eps(t_event))] => (sample_affect!, [], [sys_out.κ, sys_out.jcn, sys_in.TAN_spikes], [])
    push!(bc.discrete_callbacks, cb)

    eq = sys_in.jcn ~ w*sys_in.TAN_spikes

    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    discr_out::Matrisome,
    discr_in::Matrisome;
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    t_event = get_event_time(kwargs, nameof(discr_out), nameof(discr_in))
    cb = [t_event] => [sys_in.H ~ ifelse(sys_out.H*sys_out.jcn > sys_in.H*sys_in.jcn, 0, 1)]
    push!(bc.discrete_callbacks, cb)
end

function (bc::BloxConnector)(
    stim::ImageStimulus,
    neuron::Union{HHNeuronExciBlox, HHNeuronInhibBlox};
    kwargs...
)   
    sys_out = get_namespaced_sys(stim)
    sys_in = get_namespaced_sys(neuron)

    pixels = namespace_parameters(sys_out)

    w = generate_weight_param(stim, neuron; kwargs...)
    push!(bc.weights, w)
    # No check for kwargs[:learning_rule] here. 
    # The connection from stimulus is conceptual, the weight can not be updated.

    eq = sys_in.I_in ~ w * pixels[stim.current_pixel]

    stim.current_pixel += 1
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    stim::ImageStimulus,
    cb::CorticalBlox;
    kwargs...
)
    neurons = get_exci_neurons(cb)

    for neuron in neurons
        bc(stim, neuron; kwargs...)
    end
end

(bc::BloxConnector)(blox, as::AbstractActionSelection; kwargs...) = nothing

function connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, get_matrisome(str1), get_matrisome(str2))
end

function connect_action_selection!(as::AbstractActionSelection, matr1::Matrisome, matr2::Matrisome)
    sys1 = get_namespaced_sys(matr1)
    sys2 = get_namespaced_sys(matr2)

    as.competitor_states = [sys1.ρ_, sys2.ρ_] #HACK : accessing values of rho at a specific time after the simulation
    #as.competitor_params = [sys1.H, sys2.H]
end

# Connects spiking neuron to another spiking neuron
# None of these neurons have delays yet
function (bc::BloxConnector)(
    bloxout::AbstractNeuronBlox, 
    bloxin::AbstractNeuronBlox; 
    kwargs...
)

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    cr = get_connection_rule(kwargs, bloxout, bloxin, w)
    eq = sys_in.jcn ~ cr
    
    accumulate_equation!(bc, eq)
end

# Connects a neural mass as a driving input to a spiking neuron
# Should be used with care because units will be strange (NMM typically outputs voltage but neuron inputs are typically currents)
function (bc::BloxConnector)(
    bloxout::AbstractNeuronBlox, 
    bloxin::NeuralMassBlox; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    if typeof(bloxout.output) == Num
        x = namespace_expr(bloxout.output, sys_out)
        eq = sys_in.jcn ~ x*w
    else
        @variables t
        delay = get_delay(kwargs, nameof(bloxout), nameof(bloxin))
        τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
        τ = only(@parameters $(τ_name)=delay)
        push!(bc.delays, τ)

        x = namespace_expr(bloxout.output, sys_out)
        eq = sys_in.jcn ~ x(t-τ)*w
    end
    
    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    bloxout::LIFExciNeuron, 
    bloxin::Union{LIFExciNeuron, LIFInhNeuron}; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w * sys_in.S_AMPA * sys_in.g_AMPA * (sys_in.V - sys_in.V_E) + 
                    w * sys_in.S_NMDA * sys_in.g_NMDA * (sys_in.V - sys_in.V_E) / 
                    (1 + sys_in.Mg * exp(-0.062 * sys_in.V) / 3.57)

    accumulate_equation!(bc, eq)
    
    accumulate_spike_affect_states!(bc, nameof(sys_out), [sys_in.S_AMPA, sys_in.x])

    cb = [sys_out.V ~ sys_out.θ] => (
        LIF_spike_affect!, 
        [sys_out.V, sys_in.S_AMPA, sys_in.x], 
        [sys_out.V_reset, sys_out.t_refract_duration, sys_out.t_refract_end, sys_out.is_refractory], 
        [], 
        nothing
    )
    
    push!(bc.continuous_callbacks, cb)
end

function (bc::BloxConnector)(
    bloxout::LIFInhNeuron, 
    bloxin::Union{LIFExciNeuron, LIFInhNeuron}; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w * sys_in.S_GABA * sys_in.g_GABA * (sys_in.V - sys_in.V_I) 
                    
    accumulate_equation!(bc, eq)

    cb = [sys_out.V ~ sys_out.θ] => [sys_in.S_GABA ~ sys_in.S_GABA + 1]
    push!(bc.continuous_callbacks, cb)
end

function (bc::BloxConnector)(
    stim::PoissonSpikeTrain, 
    neuron::Union{LIFExciNeuron, LIFInhNeuron};
    kwargs...
)
    sys_in = get_namespaced_sys(neuron)

    w = generate_weight_param(stim, neuron; kwargs...)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w * sys_in.S_AMPA * sys_in.g_AMPA_external * (sys_in.V - sys_in.V_E) 
                    
    accumulate_equation!(bc, eq)

    t_spikes = generate_spike_times(stim)

    cb = t_spikes => [sys_in.S_AMPA ~ sys_in.S_AMPA + 1]
    # TO DO : Consider generating spikes during simulation
    # to make PoissonSpikeTrain independent of `t_span` of the simulation.
    # something like : 
    # discrete_event = t > -Inf => (generate_spike, [sys_in.S_AMPA], [stim.relevant_params...], [], nothing) 
    # This way we need to resolve the case of multiple spikes potentially being generated within a single integrator step.

    push!(bc.discrete_callbacks, cb)
end

function (bc::BloxConnector)(
    stim::PoissonSpikeTrain, 
    cb::Union{LIFExciCircuitBlox, LIFInhCircuitBlox};
    kwargs...
)
    neurons_in = get_neurons(cb)

    for neuron in neurons_in
        bc(stim, neuron; kwargs...)
    end
end
