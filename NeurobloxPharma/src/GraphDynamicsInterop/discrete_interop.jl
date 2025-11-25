
#-------------------------
# Matrisome

function GraphDynamics.to_subsystem(blox::Matrisome)
    # Default state initial values
    states = SubsystemStates{Matrisome}()
    # Parameter values
    (; system, t_event) = blox
    name = namespaced_nameof(blox)
    @recursive_getdefaults(system, H, TAN_spikes, jcn, jcn_, H_)
    
    params = SubsystemParams{Matrisome}(; name, H, TAN_spikes, jcn_snapshot=float(jcn_), jcn_t_block=float(jcn), H_snapshot=H_,  t_event)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{Matrisome}) = (; jcn=0.0)

GraphDynamics.subsystem_differential_requires_inputs(::Type{Matrisome}) = false
function GraphDynamics.apply_subsystem_differential!(_, m::Subsystem{Matrisome}, _, _)
    nothing
end
function GraphDynamics.computed_properties(m::Subsystem{Matrisome})
    H_learning((;H)) = H
    ρ_snapshot((;H_snapshot, jcn_snapshot)) = H_snapshot * jcn_snapshot
    ρ((;H, jcn_t_block)) = H * jcn_t_block

    # For compatibility with connect_action_selection!
    ρ_(sys) = ρ_snapshot(sys)
    H_((; H_snapshot)) = H_snapshot
    jcn_((; jcn_snapshot)) = jcn_snapshot
    (;H_learning, ρ_snapshot, ρ, ρ_, H_, jcn_)
end

GraphDynamics.event_times(m::Subsystem{Matrisome}) = m.t_event + √(eps(m.t_event))
GraphDynamics.has_discrete_events(::Type{Matrisome}) = true
GraphDynamics.discrete_events_require_inputs(::Type{Matrisome}) = false
function GraphDynamics.discrete_event_condition(m::Subsystem{Matrisome}, t, _)
    t == m.t_event + √(eps(m.t_event))
end
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{Matrisome}, _)
    # recording the values of jcn_t_block and H at the event time in the parameters jcn_ and H_
    params = get_params(s)
    @reset params.H_snapshot = s.H
    @reset params.jcn_snapshot = s.jcn_t_block 
    vparams[] = params
    nothing
end

#-------------------------
# Striosome

function GraphDynamics.to_subsystem(blox::Striosome)
    # Default state initial values
    states = SubsystemStates{Striosome}()
    # Parameter values
    (; system,) = blox
    name = namespaced_nameof(blox)
    @recursive_getdefaults(system, H, jcn)
    params = SubsystemParams{Striosome}(; name, H, jcn_t_block=float(jcn))
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{Striosome}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{Striosome}) = false
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{Striosome}, (;jcn), _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{Striosome})
    H_learning((;H)) = H
    ρ((;H, jcn_t_block)) = H * jcn_t_block
    (;H_learning, ρ)
end

#-------------------------
# TAN

function GraphDynamics.to_subsystem(blox::TAN)
    # Default state initial values
    states = SubsystemStates{TAN}()
    # Parameter values
    (; system,) = blox
    name = namespaced_nameof(blox)
    @recursive_getdefaults(system, κ, λ, jcn, rng)
    params = SubsystemParams{TAN}(; name, κ, λ, jcn_t_block=float(jcn), rng)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{TAN}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{TAN}) = false
function GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{TAN}, _, _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{TAN})
    R((;κ, λ, jcn_t_block)) = min(κ, κ/(λ*jcn_t_block + sqrt(eps())))
    (;R)
end

#-------------------------
# SNc

function GraphDynamics.to_subsystem(blox::SNc)
    # Default state initial values
    states = SubsystemStates{SNc}()
    # Parameter values
    (; system, t_event, κ_DA) = blox
    name = namespaced_nameof(blox)
    @recursive_getdefaults(system, κ, λ, jcn, jcn_)
    params = SubsystemParams{SNc}(; name, κ_DA, λ_DA=λ, κ, λ, jcn_t_block=float(jcn), jcn_snapshot=float(jcn_), t_event)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{SNc}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{SNc}) = false
function GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{SNc}, _, _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{SNc})
    R_snapshot((;κ, λ, jcn_snapshot)) = min(κ, κ/(λ*jcn_snapshot + sqrt(eps())))
    R((;κ, λ, jcn_t_block)) = min(κ, κ/(λ*jcn_t_block + sqrt(eps())))
    R_(x) = R_snapshot(x)
    (;R_snapshot, R, R_)
end

function get_modulator_state(s::Subsystem{SNc})
    Symbol(s.name, :₊R_snapshot)
end
function get_modulator_state(s::SNc)
    Symbol(s.name, :₊R_snapshot)
end
(b::Subsystem{SNc})(R_DA) = R_DA
GraphDynamics.has_discrete_events(::Type{SNc}) = true
#GraphDynamics.discrete_events_require_inputs(::Type{SNc}) = true
function GraphDynamics.discrete_event_condition((;t_event,)::Subsystem{SNc}, t, _)
    t == t_event
end
GraphDynamics.event_times((;t_event)::Subsystem{SNc}) = t_event
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{SNc}, _)
    # recording the values of jcn_t_block at the event time in the parameters jcn_snapshot
    params = get_params(s)
    vparams[] = @set params.jcn_snapshot = params.jcn_t_block
    nothing
end
