# blox that output time-series of certain frequency and phase properties
"""
phase_inter is creating a function that interpolates the phase
data for any time given
phase_inter has the following parameters:
    phase_range:  a range, e.g. 0:0.1:50 which should reflect the time points of the data
    phase_data: phase at equidistant time points
and returns:
    an function that returns an interpolated phase for t in range
"""
function phase_inter(phase_range,phase_data)
    if typeof(phase_data) == Matrix{Float64}
        return cubic_spline_interpolation(phase_range,vec(phase_data))
    end
    return cubic_spline_interpolation(phase_range,phase_data)
end

"""
phase_cos_blox is creating a cos with angular frequency ω and variable phase
phase_inter has the following parameters:
    ω: angular frequency
    t: time
    phase_inter: a function that returns phase as a function of time
and returns:
    the resulting value

Usage:
    phase_int = phase_inter(0:0.1:50,phase_data)
    phase_out(t) = phase_cos_blox(0.1,t,phase_int)
    which is now a function of time and can be used in an input blox
    you can also use the dot operator to calculate time-series
    signal = phase_out.(collect(0:0.01:50))
"""
function phase_cos_blox(ω,t,phase_inter::F) where F
    if (t in phase_inter.itp.ranges[1])
        phase = phase_inter(t)
    else
        phase = 0.0
    end
    return cos(ω*t + phase)
end

"""
phase_sin_blox is creating a sin with angular frequency ω and variable phase
phase_inter has the following parameters:
    ω: angular frequency
    t: time
    phase_inter: a function that returns phase as a function of time
and returns:
    the resulting value

Usage:
    phase_int = phase_inter(0:0.1:50,phase_data)
    phase_out(t) = phase_sin_blox(0.1,t,phase_int)
    which is now a function of time and can be used in an input blox
    you can also use the dot operator to calculate time-series
    signal = phase_out.(collect(0:0.01:50))
"""
function phase_sin_blox(ω,t,phase_inter::F) where F
    if (t in phase_inter.itp.ranges[1])
        phase = phase_inter(t)
    else
        phase = 0.0
    end
    return sin(ω*t + phase)
end

mutable struct phase_cos_signal
    # all parameters are Num as to allow symbolic expressions
    ω::Num
    signal_connector::Num
    signal_function::Function
    function phase_cos_signal(;name, ω=0.0, inp::Num, phase_range, phase_data)
        phase_int = phase_inter(phase_range,phase_data)
        signal(t) = phase_cos_blox(ω,t,phase_int)
        new(ω, inp, signal)
    end
end
