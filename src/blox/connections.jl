mutable struct BloxConnector
    eqs::Vector{Equation}
    weights::Vector{Num}
    delays::Vector{Num}

    BloxConnector() = new(Equation[], Num[])

    function BloxConnector(bloxs)
        eqs = reduce(vcat, input_equations.(bloxs)) 
        weights = reduce(vcat, weight_parameters.(bloxs))

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs + eq.rhs
end
    end
end
"""
    Helper to merge delays and weights into a single vector
"""
function params(bc::BloxConnector)
    return vcat(bc.weights, bc.delays)
end

function (bc::BloxConnector)(
    HH_out::Union{HHNeuronExciBlox, HHNeuronInhibBlox}, 
    HH_in::Union{HHNeuronExciBlox, HHNeuronInhibBlox}; 
    kwargs...
)
    sys_out = get_namespaced_sys(HH_out)
    sys_in = get_namespaced_sys(HH_in)

    weight = get_weight(kwargs, nameof(HH_out), nameof(HH_in))
    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    eq = sys_in.I_syn ~ w * sys_out.G * (sys_in.V - sys_out.E_syn)
    
    accumulate_equation!(bc, eq)
end

# function (bc::BloxConnector)(
#     jc::JansenRit, 
#     bloxin; 
#     weight = 1,
#     delay = 0
# )
#     # Need t for the delay term
#     @variables t

#     sys_out = get_namespaced_sys(jc)
#     sys_in = get_namespaced_sys(bloxin)

#     # Define & accumulate delay parameter
#     τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
#     τ = only(@parameters $(τ_name)=delay)
#     push!(bc.delays, τ)

#     w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
#     w = only(@parameters $(w_name)=weight)
#     push!(bc.weights, w)

#     x = namespace_expr(jc.connector, sys_out, nameof(sys_out))
#     eq = sys_in.jcn ~ x(t-τ)*w
    
#     accumulate_equation!(bc, eq)
# end

function (bc::BloxConnector)(
    bloxout::NeuralMassBlox, 
    bloxin::NeuralMassBlox; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    weight = get_weight(kwargs, nameof(bloxout), nameof(bloxin))
    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    if typeof(bloxout.output) == Num
        x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
        eq = sys_in.jcn ~ x*w
    else
        @variables t
        delay = get_delay(kwargs, nameof(bloxout), nameof(bloxin))
        τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
        τ = only(@parameters $(τ_name)=delay)
        push!(bc.delays, τ)

        x = namespace_expr(bloxout.output, sys_out, nameof(sys_out))
        eq = sys_in.jcn ~ x(t-τ)*w
    end
    
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
                bc(neuron_presyn, neuron_postsyn; weight, delay)
            end
        end
    end
end

function (bc::BloxConnector)(
    cb_out::Union{CorticalBlox,STN,Thalamus},
    cb_in::Union{CorticalBlox,STN,Thalamus};
    kwargs...
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)

    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))

    outgoing_connections = zeros(Int, length(neurons_out))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)

        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; weight)
        end
        outgoing_connections[idx] .+= 1
    end
end

#connection from excitatory to inhibitory neural blocks
function (bc::BloxConnector)(
    cb_out::Union{CorticalBlox,STN,Thalamus},
    cb_in::Union{GPi, GPe};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_exci_neurons(cb_out)


function (bc::BloxConnector)(
    cb::CorticalBlox,
    str::Striatum;
    kwargs...
)
        end
        outgoing_connections[idx] .+= 1
    end
end

function (bc::BloxConnector)(
    cb_out::Union{Striatum, GPi, GPe},
    cb_in::Union{CorticalBlox,STN,Thalamus};
    kwargs...
)
    neurons_in = get_exci_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))

    outgoing_connections = zeros(Int, length(neurons_out))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)

        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; weight)
        end
        outgoing_connections[idx] .+= 1
    end
end

#connection from inhibitory to inhibitory neural blocks
function (bc::BloxConnector)(
    cb_out::Union{Striatum, GPi, GPe},
    cb_in::Union{Striatum, GPi, GPe};
    kwargs...
)
    neurons_in = get_inh_neurons(cb_in)
    neurons_out = get_inh_neurons(cb_out)

    N_connects =  density * length(neurons_in) * length(neurons_out)
    out_degree = Int(ceil(N_connects / length(neurons_out)))
    in_degree =  Int(ceil(N_connects / length(neurons_in)))

    outgoing_connections = zeros(Int, length(neurons_out))
    for neuron_postsyn in neurons_in
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)

        for neuron_presyn in neurons_out[idx]
            bc(neuron_presyn, neuron_postsyn; weight)
        end
        outgoing_connections[idx] .+= 1
    end
end

function (bc::BloxConnector)(
    stim::ImageStimulus,
    neuron::Union{HHNeuronExciBlox, HHNeuronInhibBlox};
    kwargs...
)   
    sys_out = get_namespaced_sys(stim)
    sys_in = get_namespaced_sys(neuron)

    dots = namespace_variables(sys_out)

    weight = get_weight(kwargs, nameof(stim), nameof(neuron))
    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    eq = sys_in.I_in ~ w * dots[stim.current_pixel]
    eq = sys_in.I_in ~ w * dots[stim.current_pixel]

    stim.current_pixel += 1
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

function (bc::BloxConnector)(
    neuron::HHNeuronExciBlox,
    discr::AbstractDiscrete;
    kwargs...
)
    sys_out = get_namespaced_sys(neuron)
    sys_in = get_namespaced_sys(discr)

    weight = get_weight(kwargs, nameof(neuron), nameof(discr))
    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w*sys_out.spikes_window

    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    cb::CorticalBlox,
    discr::AbstractDiscrete;
    kwargs...
)
    neurons = get_exci_neurons(cb)

    for neuron in neurons
        bc(neuron, discr; kwargs...)
    end
end

function (bc::BloxConnector)(
    discr_out::DiscreteSpikes,
    discr_in::DiscreteInvSpikes;
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    weight = get_weight(kwargs, nameof(discr_out), nameof(discr_in))

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w*sys_out.ρ

    accumulate_equation!(bc, eq)
end

sample_poisson(λ) = rand(Poisson(λ))
@register_symbolic sample_poisson(λ)

function (bc::BloxConnector)(
    discr_out::DiscreteInvSpikes,
    discr_in::DiscreteSpikes;
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    weight = get_weight(kwargs, nameof(discr_out), nameof(discr_in))

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.weights, w)

    eq = sys_in.jcn ~ w*sample_poisson(sys_out.R)

    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    discr_out::DiscreteSpikes,
    discr_in::DiscreteSpikes;
    kwargs...
)
    sys_out = get_namespaced_sys(discr_out)
    sys_in = get_namespaced_sys(discr_in)

    t_event = get_event_time(kwargs, nameof(discr_out), nameof(discr_in))
    cb = t_event => [sys_in.H ~ IfElse.ifelse(sys_out.ρ > sys_in.ρ, 0, 1)]
    push!(bc.events, cb)
end

function (bc::BloxConnector)(
    policy::AbstractActionSelection,
    blox1,
    blox2;
    kwargs...
)
    sys1 = get_namespaced_sys(blox1)
    sys2 = get_namespaced_sys(blox2)

    policy.competitor_states = [sys1.jcn, sys2.jcn]
end
