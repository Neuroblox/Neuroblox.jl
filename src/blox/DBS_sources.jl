struct DBS <: StimulusBlox
    params::Vector{Num}
    odesystem::ODESystem
    namespace::Union{Symbol, Nothing}
    stimulus::Function

    function DBS(;
        name,
        namespace=nothing,
        frequency=0.130,
        amplitude=2.5,
        pulse_width=0.066,
        offset=0.0,
        start_time=0.0,
        smooth=1e-4
    )

        if smooth == 0
            stimulus = t -> square(t, frequency, amplitude, offset, start_time, pulse_width)
        else
            stimulus = t -> square(t, frequency, amplitude, offset, start_time, pulse_width, smooth)
        end

        p = paramscoping(
            frequency=frequency,
            amplitude=amplitude,
            pulse_width=pulse_width,
            offset=offset,
            start_time=start_time
        )

        sts = @variables u(t) [output = true]

        eqs = [u ~ stimulus(t)]
        sys = System(eqs, t, sts, p; name=name)
        
        new(p, sys, namespace, stimulus)
    end
end


struct DBSProtocol <: StimulusBlox
    params::Vector{Num}
    odesystem::ODESystem
    namespace::Union{Symbol, Nothing}
    stimulus::Function
    pulses_per_burst::Int
    bursts_per_block::Int

    function DBSProtocol(;
        name,
        namespace=nothing,
        frequency=130.0,
        amplitude=100.0,
        pulse_width=0.15,
        offset=0.0,
        pulses_per_burst=10,
        bursts_per_block=12,
        pre_block_time=200.0,
        inter_burst_time=200.0,
        pulse_start_time=0.0,
        smooth=1e-4
    )
        # Promote numerical inputs
        frequency, amplitude, pulse_width, offset, pre_block_time, inter_burst_time = 
            promote(frequency, amplitude, pulse_width, offset, pre_block_time, inter_burst_time)
        
        # Calculate timing parameters
        pulse_period = 1/frequency  # Time between pulses in ms
        burst_duration = pulse_period * pulses_per_burst  # Duration of each burst
        
        # Create closure with all necessary parameters
        let frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, 
            offset=offset, smooth=smooth, pre_block_time=pre_block_time,
            burst_duration=burst_duration, inter_burst_time=inter_burst_time,
            bursts_per_block=bursts_per_block, pulse_start_time=pulse_start_time

            protocol_stimulus = t -> begin
                # Adjust time relative to protocol start and pre-block period
                t_adjusted = t - pre_block_time
                burst_plus_gap = burst_duration + inter_burst_time
                current_burst = floor(t_adjusted / burst_plus_gap)
                t_within_burst_cycle = t_adjusted - current_burst * burst_plus_gap

                # Nested (compatible with symbolics) conditions for protocol timing:
                ifelse(t < pre_block_time,  # Before pre-block period
                    offset,
                    ifelse(current_burst >= bursts_per_block,  # After all bursts complete
                        offset,
                        ifelse(t_within_burst_cycle >= burst_duration - pulse_width/2,  # Between bursts
                            offset,
                            ifelse(smooth == 0,  # Generate pulse
                                square(t_within_burst_cycle, frequency, amplitude, offset, pulse_start_time, pulse_width),
                                square(t_within_burst_cycle, frequency, amplitude, offset, pulse_start_time, pulse_width, smooth)
                            )
                        )
                    )
                )
            end

            p = paramscoping(
                frequency=frequency,
                amplitude=amplitude,
                pulse_width=pulse_width,
                offset=offset,
                pre_block_time=pre_block_time,
                inter_burst_time=inter_burst_time
            )

            sts = @variables u(t) [output = true]
            eqs = [u ~ protocol_stimulus(t)]
            sys = System(eqs, t, sts, p; name=name)
            
            new(p, sys, namespace, protocol_stimulus, pulses_per_burst, bursts_per_block)
        end
    end
end

function sawtooth(t, f, offset)
    f * (t - offset) - floor(f * (t - offset))
end

# Smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width, δ)
    invδ = 1 / δ
    pulse_width_fraction = pulse_width * f
    threshold = 1 - 2 * pulse_width_fraction
    amp_half = 0.5 * amplitude
    start_time = start_time + 0.5 * pulse_width

    saw = sawtooth(t, f, start_time)
    triangle_wave = 4 * abs(saw - 0.5) - 1
    y = amp_half * (1 + tanh(invδ * (triangle_wave - threshold))) + offset

    return y
end

# Non-smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width)

    saw1 = sawtooth(t - start_time, f, pulse_width)
    saw2 = sawtooth(t - start_time, f, 0)
    saw3 = sawtooth(-start_time, f, pulse_width)
    saw4 = sawtooth(-start_time, f, 0)
    
    y = amplitude * (saw1 - saw2 - saw3 + saw4) + offset

    return y
end

function detect_transitions(t, signal::Vector{T}; atol=0) where T <: AbstractFloat
    low = minimum(signal)
    high = maximum(signal)

    # Get indexes when the signal is approximately equal to its low and high values
    low_inds = isapprox.(signal, low; atol=atol)
    high_inds = isapprox.(signal, high; atol=atol)

    # Detect each type of transitions
    trans_inds_1 = diff(low_inds) .== 1 
    trans_inds_2 = diff(low_inds) .== -1 
    trans_inds_3 = diff(high_inds) .== 1 
    trans_inds_4 = diff(high_inds) .== -1 
    circshift!(trans_inds_1, -1)
    circshift!(trans_inds_3, -1)

    # Combine all transition
    transitions_inds = trans_inds_1 .| trans_inds_2 .| trans_inds_3 .| trans_inds_4
    pushfirst!(transitions_inds, false)

    return transitions_inds
end

function compute_transition_times(stimulus::Function, f , dt, tspan, start_time, pulse_width; atol=0)
    period = 1 / f
    n_periods = floor((tspan[end] - start_time) / period)

    # Detect single pulse transition points
    t = (start_time + 0.5 * period):dt:(start_time + 1.5 * period)
    s = stimulus.(t)
    transitions_inds = detect_transitions(t, s, atol=atol)
    single_pulse = t[transitions_inds]

    # Calculate pulse times across all periods
    period_offsets = (-1:n_periods+1) * period
    pulses = single_pulse .+ period_offsets'
    transition_times = vec(pulses)

    # Filter estimated times within the actual time range
    inds = (transition_times .>= tspan[1]) .& (transition_times .<= tspan[end])
    
    return transition_times[inds]
end

function compute_transition_values(transition_times, t, signal)

    # Ensure transition_points are within the range of t, assuming both are ordered
    @assert begin
        t[1] <= transition_times[1]
        transition_times[end] <= t[end] 
    end "Transition points must be within the range of t"

    # Find the indices of the closest time points
    indices = searchsortedfirst.(Ref(t), transition_times)
    transition_values = signal[indices]
    
    return transition_values
end

function get_protocol_duration(dbs::DBSProtocol)
    # Access parameters in correct order based on paramscoping
    frequency = ModelingToolkit.getdefault(dbs.params[1])
    pre_block_time = ModelingToolkit.getdefault(dbs.params[5])
    inter_burst_time = ModelingToolkit.getdefault(dbs.params[6])
    
    # Calculate timing components
    pulse_period = 1/frequency
    burst_duration = dbs.pulses_per_burst * pulse_period
    block_duration = dbs.bursts_per_block * (burst_duration + inter_burst_time) - inter_burst_time
    
    return convert(Float64, pre_block_time + block_duration)
end