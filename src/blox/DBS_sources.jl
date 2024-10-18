struct DBS <: StimulusBlox
    params::Vector{Num}
    odesystem::ODESystem
    namespace::Union{Symbol, Nothing}
    stimulus::Function

    function DBS(;
        name,
        namespace=nothing,
        frequency=130.0,
        amplitude=100.0,
        pulse_width=0.15,
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

function detect_transitions(t, signal::Vector{T}; return_vals=false, atol=0) where T <: AbstractFloat
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

    if return_vals
        return t[transitions_inds], signal[transitions_inds]
    else
        return t[transitions_inds]
    end
end

function compute_transition_times(stimulus::Function, f , dt, tspan, start_time, pulse_width; atol=0)
    period = 1 / f
    n_periods = floor((tspan[end] - start_time) / period)

    # Detect single pulse transition points
    t = (start_time + 0.5 * period):dt:(start_time + 1.5 * period)
    s = stimulus.(t)
    single_pulse = detect_transitions(t, s; return_vals=false, atol=atol)

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