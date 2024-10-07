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
        
        # Calculate duty cycle from pulse width and frequency
        duty_cycle = pulse_width * frequency

        sts = @variables u(t)=offset [output = true]
        # eqs = [u ~ Blocks.smooth_square(t, smooth, frequency, amplitude, offset, start_time)]
        eqs = [u ~ smooth_square(t, smooth, frequency, amplitude, offset, start_time, duty_cycle)]

        sys = System(eqs, t, sts, p; name=name)
        
        new(p, sys, namespace)
    end
end


# based on the ModelingToolkitStandardLibrary source block,
# modified for allowing to set the pulse width

function smooth_square(x, δ, f, amplitude, offset, start_time, pulse_width)
    θ = (x - start_time) * f        # normalized time
    φ = θ - floor(θ)                # fractional part of θ, in [0,1)

    rise = atan((φ) / δ) / π + 0.5
    fall = atan((φ - pulse_width) / δ) / π + 0.5
    y = offset + amplitude * (rise - fall)
    y = y*smooth_step(x, δ, one(x), zero(x), start_time)

    return y
end

function smooth_step(x, δ, height, offset, start_time)
    offset .+ height .* (atan((x .- start_time) ./ δ) ./ π .+ 0.5)
end