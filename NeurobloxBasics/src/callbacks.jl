function generate_discrete_callbacks(blox::AbstractSpikeSource, bc::Connector; t_block = missing)
    sa = spike_affects(bc)
    name_blox = namespaced_nameof(blox)
  
    if haskey(sa, name_blox)
        eqs = sa[name_blox]

        cb = map(eqs) do eq
            # TO DO : Consider generating spikes during simulation
            # to make PoissonSpikeTrain independent of `t_span` of the simulation.
            # something like : 
            # discrete_event = t > -Inf => (generate_spike, [sys_dest.S_AMPA], [stim.relevant_params...], [], nothing) 
            # This way we need to resolve the case of multiple spikes potentially being generated within a single integrator step.
            t_spikes = generate_spike_times(blox)
            t_spikes => to_vector(eq)
        end
        
        return cb
    end
end

function LIF_spike_affect!(integ, u, p, ctx)
    integ.u[u[1]] = integ.p[p[1]]

    t_refract_end = integ.t + integ.p[p[2]]
    integ.p[p[3]] = t_refract_end

    integ.p[p[4]] = 1

    SciMLBase.add_tstop!(integ, t_refract_end)

    c = 1
    for i in eachindex(u)[2:end]
        integ.u[u[i]] += integ.p[p[c + 4]]
        c += 1
    end
end

function generate_discrete_callbacks(blox::Union{LIFExciNeuron, LIFInhNeuron}, bc::Connector; t_block = missing)
    sa = spike_affects(bc)
    name_blox = namespaced_nameof(blox)
    sys = get_namespaced_sys(blox)

    states_affect = get_states_spikes_affect(sa, name_blox)
    params_affect = get_params_spikes_affect(sa, name_blox)

    param_pairs = make_unique_param_pairs(params_affect)

    ps = vcat([
        sys.V_reset => Symbol(sys.V_reset), 
        sys.t_refract_duration => Symbol(sys.t_refract_duration), 
        sys.t_refract_end => Symbol(sys.t_refract_end), 
        sys.is_refractory => Symbol(sys.is_refractory)
    ], param_pairs)
    
    cb = (sys.V > sys.θ) => (
        LIF_spike_affect!, 
        vcat(sys.V, states_affect), 
        ps, 
        [], 
        nothing
    )

    return cb
end
