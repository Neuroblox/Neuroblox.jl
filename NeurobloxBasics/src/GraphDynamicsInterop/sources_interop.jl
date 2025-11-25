
function GraphDynamics.to_subsystem(s::PoissonSpikeTrain)
    states = SubsystemStates{PoissonSpikeTrain, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{PoissonSpikeTrain}((;))
    Subsystem(states, params)
end
GraphDynamics.initialize_input(s::Subsystem{PoissonSpikeTrain}) = (;)
GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{PoissonSpikeTrain}, _, _) = nothing
GraphDynamics.subsystem_differential_requires_inputs(::Type{PoissonSpikeTrain}) = false
