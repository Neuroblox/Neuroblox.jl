


function GraphDynamics.to_subsystem(s::PoissonSpikeTrain)
    states = SubsystemStates{PoissonSpikeTrain, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{PoissonSpikeTrain}((;))
    Subsystem(states, params)
end
GraphDynamics.initialize_input(s::Subsystem{PoissonSpikeTrain}) = (;)
GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{PoissonSpikeTrain}, _, _) = nothing
GraphDynamics.subsystem_differential_requires_inputs(::Type{PoissonSpikeTrain}) = false


#--------------------------------------------

function GraphDynamics.to_subsystem(s::ImageStimulus)
    states = SubsystemStates{ImageStimulus}()
    (; name, IMG, category, t_stimulus, t_pause, N_pixels, N_stimuli) = s
    params = SubsystemParams{ImageStimulus}(
        ;name, current_image=IMG[:,1], IMG, category, t_stimulus, t_pause, N_pixels, N_stimuli
    )
    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{ImageStimulus}) = (;) # ImageStimulus has no inputs!
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{ImageStimulus}, jcn, t)
    nothing
end

GraphDynamics.has_discrete_events(::Type{ImageStimulus}) = true
GraphDynamics.event_times((;t_stimulus)::Subsystem{ImageStimulus}) = t_stimulus
GraphDynamics.discrete_event_condition(s::Subsystem{ImageStimulus}, t, _) = s.t_stimulus == t
function GraphDynamics.apply_discrete_event!(integrator, sview, pview, s::Subsystem{ImageStimulus}, _)
    # zero out the current image
    s.current_image .= 0.0
    nothing
end

#-------------------------------

# function GraphDynamics.to_subsystem(s::PulsesInput)
#     states = SubsystemStates{PulsesInput}()
#     (;name, times_on, times_off, pulse_switch, base_line, pulse_amp) = s
#     I = base_line
#     params = SubsystemParams{PulsesInput}(;name, times_on, times_off, pulse_switch, base_line, pulse_amp, I)
#     Subsystem(states, params)
# end
# GraphDynamics.initialize_input(s::Subsystem{PulsesInput}) = (;)
# function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{PulsesInput}, jcn, t)
#     nothing
# end

# GraphDynamics.has_discrete_events(::Type{PulsesInput}) = true
# GraphDynamics.event_times((;times_on, times_off)::Subsystem{PulsesInput}) = [times_on; times_off]
# GraphDynamics.discrete_event_condition(s::Subsystem{PulsesInput}, t, _) = (t ∈ s.times_on || t ∈ s.times_off)
# function GraphDynamics.apply_discrete_event!(integrator, sview, pview, s::Subsystem{PulsesInput}, _)
#     (;t) = integrator
#     (;times_on, times_off, pulse_amp, pulse_switch, base_line) = s
#     params = pview[]
#     for i ∈ eachindex(times_on, pulse_switch)
#         if t == times_on[i]
#             pview[] = @reset params.I = pulse_amp * pulse_switch[i]
#         end
#     end
#     if t ∈ times_off
#         pview[] = @reset params.I = base_line
#     end
#     nothing
# end
# outputs((;I)::Subsystem{PulsesInput}) = (;I)
