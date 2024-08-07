@parameters t

# Simple input blox
mutable struct ExternalInput <: StimulusBlox
    namespace
    output::Num
    odesystem::ODESystem
    function ExternalInput(;name, I=1.0, namespace=nothing)
        sts = @variables u(t) [irreducible=true, description="ext_input"]
        eqs = [u ~ I]
        odesys = System(eqs, t, sts, []; name=name)
        new(namespace, sts[1], odesys)
    end
end

#CosineSource
mutable struct CosineSource	
    f::Num
    a::Num
    phi::Num
    offset::Num
    tstart::Num
    connector::Num
    odesystem::ODESystem
    function CosineSource(;name, f=18, a=10, phi=0, offset=0, tstart=0)
        @named source = Blocks.Cosine(frequency=f, amplitude=a, phase=phi, 		 
                        offset=offset, start_time=tstart, smooth=false)	
        new(f, a, phi, offset, tstart, source.output.u, source)
    end
end

#CosineBlox
mutable struct CosineBlox
    amplitude::Num
    frequency::Num
    phase::Num
    connector::Num
    odesystem::ODESystem
    function CosineBlox(;name, amplitude=1, frequency=20, phase=0)

        sts    = @variables jcn(t)=0.0 u(t)=0.0
        params = @parameters amplitude=amplitude frequency=frequency phase=phase

        eqs = [u ~ amplitude * cos(2 * pi * frequency * (t) + phase)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, phase, odesys.u, odesys)
    end
end

#NoisyCosineBlox
mutable struct NoisyCosineBlox
    amplitude::Num
    frequency::Num
    connector::Num
    odesystem::ODESystem
    function NoisyCosineBlox(;name, amplitude=1, frequency=20) 

        sts    = @variables  u(t)=0.0 jcn(t)=0.0
        params = @parameters amplitude=amplitude frequency=frequency

        eqs    = [u   ~ amplitude * cos(2 * pi * frequency * (t) + jcn)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, odesys.u, odesys)
    end
end

#PhaseBlox
mutable struct PhaseBlox
    connector::Num
    odesystem::ODESystem
    function PhaseBlox(;name, phase_range=0, phase_data=0) 

        data        = convert(Vector{Float64}, phase_data)
        range       = convert(Vector{Float64}, phase_range)
        phase_input = CubicSpline(data, range)

        sts         = @variables  u(t)=0.0 jcn(t)=0.0

        eqs         = [u ~ phase_input(t)]
        odesys      = ODESystem(eqs, t, sts, []; name=name)

        new(odesys.u, odesys)
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
    const odesystem
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

struct PoissonSpikeTrain <: StimulusBlox
    name
    namespace
    rate
    tspan
    prob_dt
    rng
    
    function PoissonSpikeTrain(; name, namespace, rate, tspan, prob_dt = 0.01, rng = MersenneTwister(1234))
        rate = to_vector(rate)
        tspan = to_vector(tspan)

        @assert length(rate) == length(tspan) "The number of Poisson rates need to match the number of tspan intervals."

        new(name, namespace, rate, tspan, prob_dt, rng)
    end
end

function generate_spike_times(stim::PoissonSpikeTrain)
    # This could also change to a dispatch of Random.rand()
    t_spikes = Float64[]
    for (rate, tspan) in zip(stim.rate, stim.tspan)
        # The dt step is determined by the CDF of the Exponential distribution.
        # The Exponential is the distribution of the inter-event times for Poisson-distributed events.
        # `prob_dt` determines the probability so that `P_CDF_Exponential(dt) = prob_dt` , and then we solve for dt.
        # This way we make sure that with probability `1 - prob_dt` there won't be any events within a single dt step.
        dt = - log(1 - stim.prob_dt) / (1 / rate)
        for t in range(tspan...; step = dt)
            if rand(stim.rng) < rate * dt
                push!(t_spikes, t)
            end
        end
    end

    return t_spikes
end
