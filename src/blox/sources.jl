abstract type AbstractSpikeSource <: StimulusBlox end

struct VoltageClampSource <: StimulusBlox
    namespace
    system

    function VoltageClampSource(clamp_schedule::Vector{@NamedTuple{t::T1, V::T2}}; name, namespace=nothing) where {T1, T2}
        @variables V(t)=0 [output=true]
        #@parameters V=0
        eqs = [D(V) ~ 0]

        cbs = map(clamp_schedule) do cs
            [cs.t] => [V ~ cs.V]
        end

        sys = System(eqs, t, [V], []; name=name, discrete_events=cbs)

        new(namespace, sys)
    end
end

struct ConstantInput <: StimulusBlox
    namespace
    system

    function ConstantInput(; name, namespace=nothing, I=1)
        @variables u(t) [output=true, description="ext_input"]
        @parameters I=I
        eqs = [u ~ I]
        sys = System(eqs, t, [u], [I]; name=name)

        new(namespace, sys)
    end
end

function get_sampled_data(t, t_trial::Real, t_stims::AbstractVector, pixel_data::AbstractVector)
    idx = floor(Int, t / t_trial) + 1
    
    return ifelse(
        (t >= first(t_stims[idx])) && (t <= last(t_stims[idx])), 
        pixel_data[idx], 
        0.0
    )
end

@register_symbolic get_sampled_data(t, t_trial::Real, t_stims::AbstractVector, pixel_data::AbstractVector)

mutable struct ImageStimulus <: StimulusBlox
    const namespace
    const system
    const IMG # Matrix[pixels X stimuli]
    const stim_parameters
    const category
    const t_stimulus
    const t_pause
    const N_pixels
    const N_stimuli
    current_pixel::Int

    function ImageStimulus(data::DataFrame; name, namespace, t_stimulus, t_pause)
        N_pixels = DataFrames.ncol(data[!, Not(:category)])
        N_stimuli = DataFrames.nrow(data[!, Not(:category)])

        # Append a row of zeros at the end of data so that indexing can work
        # on the final simulation time step when the index will be `nrow(data)+1`.
        d0 = DataFrame(Dict(n => 0 for n in names(data)))
        append!(data, d0)

        S = transpose(Matrix(data[!, Not(:category)]))

        t_trial = t_stimulus + t_pause
        t_stims = [
            ((i-1)*t_trial, (i-1)*t_trial + t_stimulus)
            for i in Base.OneTo(N_stimuli)
        ]
        # Append a dummy stimulation interval at the end
        # so that index is not out of bounds , similar to data above.
        push!(t_stims, (0,0))

        param_name = :u
        @parameters t
        ps = Vector{Num}(undef, N_pixels)
        reset_eqs = Vector{Equation}(undef, N_pixels)
        for i in Base.OneTo(N_pixels)
            s = Symbol(param_name, "_", i)
            ps[i] = only(@parameters $(s) = S[i,1])
            reset_eqs[i] = ps[i] ~ 0.0
        end

        cb_stop_stim = [t_stimulus] => reset_eqs
        sys = ODESystem(Equation[], t, [], ps; name, discrete_events = cb_stop_stim)
        category = data[!, :category]

        ps_namespaced = namespace_parameters(get_namespaced_sys(sys))

        new(namespace, sys, S, ps_namespaced, category, t_stimulus, t_pause, N_pixels, N_stimuli, 1)
    end

    function ImageStimulus(file::String; name, namespace, t_stimulus, t_pause)
        @assert last(split(file, '.')) == "csv" "Image file must be a CSV file."
        data = read(file, DataFrame)
        ImageStimulus(data; name, namespace, t_stimulus, t_pause)
    end
end

increment_pixel!(stim::ImageStimulus) = stim.current_pixel = mod(stim.current_pixel, stim.N_pixels) + 1

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
