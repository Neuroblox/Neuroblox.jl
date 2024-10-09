struct DBS <: StimulusBlox
    params::Vector{Num}
    odesystem::ODESystem
    namespace::Union{Symbol, Nothing}

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
        p = paramscoping(
            frequency=frequency,
            amplitude=amplitude,
            pulse_width=pulse_width,
            offset=offset,
            start_time=start_time
        )

        frequency, amplitude, pulse_width, offset, start_time = p
        sts = @variables u(t)=offset [output = true]

        if smooth == 0
            eqs = [u ~ square(t, frequency, amplitude, offset, start_time, pulse_width)]
        else
            eqs = [u ~ square(t, frequency, amplitude, offset, start_time, pulse_width, smooth)]
        end

        sys = System(eqs, t, sts, p; name=name)
        
        new(p, sys, namespace)
    end
end

function sawtooth(t, f, offset)
    f * (t - offset) .- floor.(f * (t - offset))
end

# smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width, δ)
    invδ = 1 / δ
    threshold = 1 - 2 * pulse_width
    amp_half = amplitude * 0.5
    start_time = start_time + pulse_width*0.5

    saw = sawtooth(t, f, start_time)
    triangle_wave = 4 * abs(saw - 0.5) - 1
    y = amp_half * (1 + tanh((triangle_wave - threshold) * invδ)) + offset

    return y
end

# non-smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width)

    saw1 = sawtooth(t, f, start_time + pulse_width)
    saw2 = sawtooth(t, f, start_time)
    saw3 = sawtooth(0.0, f, start_time + pulse_width)
    saw4 = sawtooth(0.0, f, start_time)
    
    y = amplitude * (saw1 - saw2 - saw3 + saw4 + 1.0) + offset
    
    return y
end