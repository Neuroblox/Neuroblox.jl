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

        sts = @variables u(t)=offset [output = true]

        eqs = [u ~ stimulus(t)]
        sys = System(eqs, t, sts, p; name=name)
        
        new(p, sys, namespace, stimulus)
    end
end

function sawtooth(t, f, offset)
    f * (t - offset) - floor(f * (t - offset))
end

# smoothed square pulses
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

# non-smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width)

    saw1 = sawtooth(t - start_time, f, pulse_width)
    saw2 = sawtooth(t - start_time, f, 0)
    saw3 = sawtooth(-start_time, f, pulse_width)
    saw4 = sawtooth(-start_time, f, 0)
    
    y = amplitude * (saw1 - saw2 - saw3 + saw4) + offset

    return y
end

# Detect the transition times of the boundaries of each pulse
# Less efficient but sometimes more robust method
function compute_transition_times(t, signal; threshold=250)
    is_high = signal .>= threshold
    transitions = findall(diff(is_high) .!= 0)
    
    transition_points = Float64[]
    
    for i in transitions
        if is_high[i]  # Transition from low to high
            last_zero = findlast(x -> x < threshold, signal[1:i])
            if !isnothing(last_zero)
                push!(transition_points, t[last_zero])
            end
            push!(transition_points, t[i+1])
        else  # Transition from high to low
            last_high = findlast(x -> x >= threshold, signal[1:i])
            if !isnothing(last_high)
                push!(transition_points, t[last_high])
            end
            push!(transition_points, t[i+1])
        end
    end
    
    # Handle the last transition if it's not caught by the loop
    if !isempty(transitions)
        last_transition = transitions[end]
        if is_high[last_transition+1]  # If the last detected transition was to high
            last_high = findlast(x -> x >= threshold, signal)
            if !isnothing(last_high) && last_high > last_transition+1
                push!(transition_points, t[last_high])
                last_zero = findnext(x -> x < threshold, signal, last_high)
                if !isnothing(last_zero)
                    push!(transition_points, t[last_zero])
                end
            end
        else  # If the last detected transition was to low
            last_zero = findlast(x -> x < threshold, signal)
            if !isnothing(last_zero) && last_zero > last_transition+1
                last_high = findlast(x -> x >= threshold, signal[1:last_zero-1])
                if !isnothing(last_high)
                    push!(transition_points, t[last_high])
                end
                push!(transition_points, t[last_zero])
            end
        end
    end
    
    return sort(unique(transition_points))
end

# Detect the transition times of the boundaries of each pulse
# Much more efficient but possibly less robust method
#
# Δ should be the solver's dt if using a square pulse without smoothing,
# or some other value accounting for smoothing otherwise
function compute_transition_times(t, f, start_time, pulse_width, dt; smooth=0)
    period = 1 / f
    n_periods = floor((t[end] - start_time) / period)

    # Define single pulse transition points
    if smooth == 0
        single_pulse = [start_time - dt, start_time, start_time + pulse_width - dt, start_time + pulse_width]
    else
        Δ = abs(log(smooth))*dt*0.1
        single_pulse = [start_time - Δ, start_time + dt + Δ*0.1, start_time + pulse_width - dt - Δ*0.1, start_time + pulse_width + Δ]
        @show single_pulse
    end

    # Calculate pulse times across all periods
    period_offsets = (0:n_periods) * period
    pulses = single_pulse .+ period_offsets'
    transition_times = vec(pulses)

    # Filter estimated times within the actual time range
    inds = (transition_times .>= t[1]) .& (transition_times .<= t[end])
    
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