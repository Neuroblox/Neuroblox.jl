mutable struct BloxConnector
    eqs::Vector{Equation}
    weights::Vector{Num}
    delays::Vector{Num}
    events
    learning_rules

    BloxConnector() = new(Equation[], Num[], Num[], Pair{Any, Vector{Equation}}[], Dict{Num, AbstractLearningRule}())

    function BloxConnector(bloxs)
        eqs = reduce(vcat, input_equations.(bloxs)) 
        weights = reduce(vcat, weight_parameters.(bloxs))
        delays = reduce(vcat, delay_parameters.(bloxs))
        events = reduce(vcat, event_callbacks.(bloxs))
        learning_rules = reduce(merge, weight_learning_rules.(bloxs))

        new(eqs, weights, delays, events, learning_rules)
    end
end

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs + eq.rhs
end

get_equations_with_parameter_lhs(bc) = filter(eq -> isparameter(eq.lhs), bc.eqs)

get_equations_with_state_lhs(bc) = filter(eq -> !isparameter(eq.lhs), bc.eqs)

function get_callbacks(bc, t_affect=missing)
    if !ismissing(t_affect)
        cbs_params = t_affect => get_equations_with_parameter_lhs(bc)

        return vcat(cbs_params, bc.events)
    else
        return bc.events
    end
end

function generate_callbacks_for_parameter_lhs(bc)
    eqs = get_equations_with_parameter_lhs(bc)
    cbs = [bc.param_update_times[eq.lhs] => eq for eq in eqs]

    return cbs
end

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

function hypergeometric_connections!(bc, neurons_out, neurons_in, name_out, name_in; kwargs...)
    density = get_density(kwargs, name_out, name_in)
    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))

    outgoing_connections = zeros(Int, length(neurons_out))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)

        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; kwargs...)
        end
        outgoing_connections[idx] .+= 1
    end
end

"""
    Helper to merge delays and weights into a single vector
"""
function params(bc::BloxConnector)
    weights = []
    for w in bc.weights
        append!(weights, Symbolics.get_variables(w))
        # if Symbolics.getdefaultval(w) isa Num
        #     p = Symbolics.get_variables(Symbolics.getdefaultval(w))
        #     append!(weights, p)
        # else
        #     append!(weights, w)
        # end
    end
    return vcat(reduce(vcat, weights), bc.delays)
    # return vcat(bc.weights, bc.delays)
end

function (bc::BloxConnector)(
    HH_out::Union{HHNeuronExciBlox, HHNeuronInhibBlox}, 
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; 
    kwargs...
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    w = generate_weight_param(HH_out, HH_in; kwargs...)
    push!(bc.weights, w)    

    if haskey(kwargs, :learning_rule)
        lr = kwargs[:learning_rule]
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
    bloxout::NeuralMassBlox, 
    bloxin::NeuralMassBlox; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        lr = kwargs[:learning_rule]
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
            w = only(@parameters $(w_name)=weight)
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

# Ok yes this is a bad dispatch but the whole compound blocks implementation is hacky and needs fixing @@
# Opening an issue to loop back to this during clean up week
function (bc::BloxConnector)(
    bloxout::CompoundNOBlox, 
    bloxin::CompoundNOBlox; 
    weight=1,
    delay=0,
    density=0.1
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
    eq = sys_in.nmm₊jcn ~ x*w
    
    accumulate_equation!(bc, eq)
end

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

    if haskey(kwargs, :learning_rule)
        lr = kwargs[:learning_rule]
        sys_matr = get_namespaced_sys(get_matrisome(str))
        maybe_set_state_post!(lr, sys_matr.H)
        kwargs = (kwargs..., learning_rule=lr)
    end

    hypergeometric_connections!(bc, neurons_out, neurons_in, nameof(cb), nameof(str); kwargs...)

    algebraic_parts = [get_matrisome(str), get_striosome(str)]

    for neuron_presyn in neurons_out
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
    discr::Union{Matrisome, Striosome};
    kwargs...
)
    sys_out = get_namespaced_sys(neuron)
    sys_in = get_namespaced_sys(discr)

    w = generate_weight_param(neuron, discr; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :learning_rule)
        lr = kwargs[:learning_rule]
        maybe_set_state_pre!(lr, sys_out.spikes_cumulative)
        maybe_set_state_post!(lr, sys_in.H)
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
    cb_matr = t_event => [sys_matr_in.H ~ IfElse.ifelse(sys_matr_out.ρ > sys_matr_in.ρ, 0, 1)]
    cb_strios = t_event => [sys_strios_in.H ~ IfElse.ifelse(sys_matr_out.ρ > sys_matr_in.ρ, 0, 1)]
    push!(bc.events, cb_matr)
    push!(bc.events, cb_strios)

    for neuron in neurons_in
        sys_neuron = get_namespaced_sys(neuron)
        # Large negative current added to shut down the Striatum spiking neurons.
        # Value is hardcoded for now, as it's more of a hack, not user option. 
        cb_neuron = t_event => [sys_neuron.I_bg ~ IfElse.ifelse(sys_matr_out.ρ > sys_matr_in.ρ, -2, 0)]
        push!(bc.events, cb_neuron)
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
        bc.learning_rules[w] = kwargs[:learning_rule]
    end

    eq = sys_in.jcn ~ w*sys_out.ρ

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
    v = rand(Poisson(u[1]))
    integ.p[1] = v
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
        bc.learning_rules[w] = kwargs[:learning_rule]
    end

    t_event = get_event_time(kwargs, nameof(discr_out), nameof(discr_in))
    cb = t_event => (sample_affect!, [sys_out.R], [sys_out.spikes_window], nothing)
    push!(bc.events, cb)

    eq = sys_in.jcn ~ w*sys_out.spikes_window

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
    cb = t_event => [sys_in.H ~ IfElse.ifelse(sys_out.ρ > sys_in.ρ, 0, 1)]
    push!(bc.events, cb)
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

    as.competitor_states = [sys1.ρ, sys2.ρ]
end