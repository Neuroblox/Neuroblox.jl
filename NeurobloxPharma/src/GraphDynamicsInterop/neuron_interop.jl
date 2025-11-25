##----------------------------------------------
## Neurons / Neural Mass
##----------------------------------------------

for sys ∈ [HHNeuronExci(name=:hhne)
           HHNeuronInhib(name=:hhni)
           HHNeuronFSI(name=:hhnfsi)
           NGNMM_theta(name=:ngnmm_theta)]
    define_neuron(sys; mod=@__MODULE__())
end

NeurobloxBase.GraphDynamicsInterop.has_t_block_event(::Type{HHNeuronExci}) = true
NeurobloxBase.GraphDynamicsInterop.is_t_block_event_time(::Type{HHNeuronExci}, key, t) = key == :t_block_late
NeurobloxBase.GraphDynamicsInterop.t_block_event_requires_inputs(::Type{HHNeuronExci}) = false
function NeurobloxBase.GraphDynamicsInterop.apply_t_block_event!(vstates, _, s::Subsystem{HHNeuronExci}, _, _)
    vstates[:spikes_window] = 0.0
end
