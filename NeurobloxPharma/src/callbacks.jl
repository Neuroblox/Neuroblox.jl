function generate_discrete_callbacks(blox::HHNeuronExci, ::Connector; t_block = missing)
    if !ismissing(t_block)
        nn = get_namespaced_sys(blox)
        eq = nn.spikes_window ~ 0
        cb_spike_reset = (t_block + 2*sqrt(eps(float(t_block)))) => [eq]
        
        return cb_spike_reset
    else
        return []
    end
end
