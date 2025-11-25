abstract type AbstractSpikeSource <: AbstractStimulus end

"""
    ConstantInput(; namespace, namespace, I)

Create a constant input current.

Inputs:
- name: Name given to ODESystem object within the blox.
- namespace: Additional namespace above name if needed for inheritance.
- I: External current input (mA).
"""
struct ConstantInput <: AbstractSimpleStimulus
    name
    namespace
    system

    function ConstantInput(; name, namespace=nothing, I=1)
        @variables u(t) [output=true, description="ext_input"]
        @parameters I=I
        eqs = [u ~ I]
        sys = System(eqs, t, [u], [I]; name=name)

        new(name, namespace, sys)
    end
end

#CosineSource
"""
    CosineSource(name, f, a, phi, offset, tstart)

Create a sinusoidally-varying current source.

Inputs:
- name: Options containing specification about deterministic.
- f: Frequency
- a: Amplitude
- phi: Phase
- offset: Constant added to cosine signal.
- tstart: Start time
"""
mutable struct CosineSource	
    f::Num
    a::Num
    phi::Num
    offset::Num
    tstart::Num
    connector::Num
    system::ODESystem
    function CosineSource(;name, f=18, a=10, phi=0, offset=0, tstart=0)
        @named source = Blocks.Cosine(frequency=f, amplitude=a, phase=phi, 		 
                        offset=offset, start_time=tstart, smooth=false)	
        new(f, a, phi, offset, tstart, source.output.u, source)
    end
end

#CosineBlox
"""
    CosineBlox(; name, amplitude, frequency, phase)
"""
mutable struct CosineBlox
    amplitude::Num
    frequency::Num
    phase::Num
    connector::Num
    system::ODESystem
    function CosineBlox(;name, amplitude=1, frequency=20, phase=0)

        sts    = @variables jcn(t) u(t)=0.0
        params = @parameters amplitude=amplitude frequency=frequency phase=phase

        eqs = [u ~ amplitude * cos(2 * pi * frequency * (t) + phase)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, phase, odesys.u, odesys)
    end
end

#NoisyCosineBlox
"""
    NoisyCosineBlox(name, amplitude, frequency)
"""
mutable struct NoisyCosineBlox
    amplitude::Num
    frequency::Num
    connector::Num
    system::ODESystem
    function NoisyCosineBlox(;name, amplitude=1, frequency=20) 

        sts    = @variables  u(t)=0.0 jcn(t)
        params = @parameters amplitude=amplitude frequency=frequency

        eqs    = [u   ~ amplitude * cos(2 * pi * frequency * (t) + jcn)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, odesys.u, odesys)
    end
end

#PhaseBlox
"""
    PhaseBlox(name, phase_range, phase_data)

Create an input that interpolates a set of data points using a cubic spline.

Arguments:
- name: Name given to ODESystem object within the blox.
- phase_range: 
- phase_data: 
"""
mutable struct PhaseBlox
    connector::Num
    system::ODESystem
    function PhaseBlox(;name, phase_range=0, phase_data=0) 

        data        = convert(Vector{Float64}, phase_data)
        range       = convert(Vector{Float64}, phase_range)
        phase_input = CubicSpline(data, range)

        sts         = @variables  u(t)=0.0 jcn(t)

        eqs         = [u ~ phase_input(t)]
        odesys      = ODESystem(eqs, t, sts, []; name=name)

        new(odesys.u, odesys)
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
struct PoissonSpikeTrain{N} <: AbstractSpikeSource
    name
    namespace
    N_trains
    rate::N
    tspan
    prob_dt
    rng
end

function PoissonSpikeTrain(rate::Union{AbstractVector{N}, N}, tspan::Union{AbstractVector{T}, T}; name, namespace=nothing, N_trains=1, prob_dt=0.01, rng=Random.GLOBAL_RNG) where {N <: Number, T <: Tuple}
    rate = to_vector(rate)
    tspan = to_vector(tspan)

    @assert length(rate) == length(tspan) "The number of Poisson rates need to match the number of tspan intervals."

    PoissonSpikeTrain(name, namespace, N_trains, rate, tspan, prob_dt, rng)
end

function PoissonSpikeTrain(rate_sampling::NamedTuple, tspan::Tuple; name, namespace=nothing, N_trains=1, prob_dt=0.01, rng=Random.GLOBAL_RNG)     
    
    PoissonSpikeTrain(name, namespace, N_trains, rate_sampling, tspan, prob_dt, rng)
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

function generate_spike_times(stim::PoissonSpikeTrain{N}) where {N <: AbstractVector}
    # This could also change to a dispatch of Random.rand()
    t_spikes = Float64[]
    for _ in Base.OneTo(stim.N_trains)
        for i in eachindex(stim.rate)        
            generate_spike_times!(t_spikes, stim.rate[i], stim.tspan[i], stim.prob_dt, stim.rng)
        end
    end

    return t_spikes
end

function generate_spike_times(stim::PoissonSpikeTrain{N}) where {N <: NamedTuple}
    # This could also change to a dispatch of Random.rand()
    
    dist_rate = stim.rate.distribution
    dt = stim.rate.dt
    rng = stim.rng
    tspan = stim.tspan
    prob_dt = stim.prob_dt

    t_spikes = Float64[]
    for _ in Base.OneTo(stim.N_trains)
        for t in range(tspan...; step = dt)
            rate = rand(rng, dist_rate)
            tspan_sample = (t, t + dt)

            generate_spike_times!(t_spikes, rate, tspan_sample, prob_dt, rng)
        end
    end
    
    return t_spikes
end

function get_ts_data(t, dt::Real, data::Array{Float64})
    idx = ceil(Int, t / dt)

    return idx > 0 ? data[idx] : 0.0
end

@register_symbolic get_ts_data(t, dt::Real, data::Array{Float64})

"""
    ARProcess(name, namespace, dt, data)
"""
mutable struct ARProcess <: AbstractSimpleStimulus
    namespace
    system
    function AR(;name, namespace=nothing, dt, data)
        sts = @variables u(t) = 0.0 [output=true]

        eq = [u ~ get_ts_data(t, dt, data)]

        sys = System(eq, t; name)

        new(namespace, sys)
    end
end
