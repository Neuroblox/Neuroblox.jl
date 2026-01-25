abstract type AbstractSpikeSource <: AbstractStimulus end

"""
    ConstantInput(; namespace, namespace, I)

Create a constant input current.

Inputs:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- I: External current input (mA).
"""
@blox struct ConstantInput(;name, namespace=nothing, I=1.0) <: AbstractSimpleStimulus
    @params I
    @states
    @inputs
    @outputs I
    @equations begin
    end
end


"""
    PoissonSpikeTrain(rate, tspan; name, namespace, N_trains, prob_dt, rng)

Create an input that generates spikes according to a Poisson process.

Inputs:
- rate: A single rate or vector of rates for the Poisson processes.
- tspan: A single timespan tuple or vector of timespan tuples.
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- N_trains: Number of generated spike trains.
- prob_dt: Timestep of the problem, used to determine the time step for the Poisson process.
- rng: Random number generator.
"""
@blox struct PoissonSpikeTrain(;name,
                               namespace=nothing,
                               N_trains,
                               rate,
                               tspan,
                               prob_dt,
                               rng=Random.default_rng()) <: AbstractSpikeSource
    # Stick this data in the extra fields, we don't want to carry it around on the GraphDynamics
    # side, it's just used for generating connection spike events.
    @extra_fields(
        N_trains=N_trains,
        rate=rate,
        tspan=tspan,
        prob_dt=prob_dt,
        rng=rng
    )
    @params
    @states
    @inputs
    @equations begin
    end
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{PoissonSpikeTrain}) = false

function PoissonSpikeTrain(rate::Union{AbstractVector{N}, N}, tspan::Union{AbstractVector{T}, T};
                           name, namespace=nothing, N_trains=1, prob_dt=0.01, rng=Random.default_rng()) where {N <: Number, T <: Tuple}
    rate = to_vector(rate)
    tspan = to_vector(tspan)

    @assert length(rate) == length(tspan) "The number of Poisson rates need to match the number of tspan intervals."
    PoissonSpikeTrain(;name, namespace, N_trains, rate, tspan, prob_dt, rng)
end

function PoissonSpikeTrain(rate::NamedTuple, tspan::Tuple;
                           name, namespace=nothing, N_trains=1, prob_dt=0.01, rng=Random.default_rng())
    PoissonSpikeTrain(;name, namespace, N_trains, rate, tspan, prob_dt, rng)
end

function generate_spike_times((; N_trains, rate, tspan, prob_dt, rng)::PoissonSpikeTrain)
    generate_spike_times(N_trains, rate, tspan, prob_dt, rng)
end

function generate_spike_times(N_trains, rate::AbstractVector, tspan, prob_dt, rng)
    # This could also change to a dispatch of Random.rand()
    t_spikes = Float64[]
    for _ in 1:N_trains
        for i in eachindex(rate)        
            generate_spike_times!(t_spikes, rate[i], tspan[i], prob_dt, rng)
        end
    end

    return t_spikes
end

function generate_spike_times(N_trains, rate::NamedTuple, tspan, prob_dt, rng)
    # This could also change to a dispatch of Random.rand()
    
    dist_rate = rate.distribution
    dt = rate.dt

    t_spikes = Float64[]
    for _ in 1:N_trains
        for t in range(tspan...; step = dt)
            rate = rand(rng, dist_rate)
            tspan_sample = (t, t + dt)

            generate_spike_times!(t_spikes, rate, tspan_sample, prob_dt, rng)
        end
    end
    
    return t_spikes
end

function generate_spike_times!(t_spikes, rate::Number, tspan, prob_dt, rng)
    # The dt step is determined by the CDF of the Exponential distribution.
    # The Exponential is the distribution of the inter-event times for Poisson-distributed events.
    # `prob_dt` determines the probability so that `P_CDF_Exponential(dt) = prob_dt` , and then we solve for dt.
    # This way we make sure that with probability `1 - prob_dt` there won't be any events within a single dt step.
    dt = map(rate) do r
        - log(1 - prob_dt) / r
    end

    for t in range(tspan...; step = dt)
        if rand(rng) < rate * dt
            push!(t_spikes, t)
        end
    end
end

