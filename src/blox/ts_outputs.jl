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
function phase_inter(phase_range,phase_data_phase)
    return CubicSplineInterpolation(phase_range,phase_data_phase)
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
    # which is now a function of time and can be used in an input blox
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
    # which is now a function of time and can be used in an input blox
"""
function phase_sin_blox(ω,t,phase_inter::F) where F
    if (t in phase_inter.itp.ranges[1])
        phase = phase_inter(t)
    else
        phase = 0.0
    end
    return sin(ω*t + phase)
end
