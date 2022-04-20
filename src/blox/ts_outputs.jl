# blox that output time-series of certain frequency and phase properties

function phase_inter(phase_range,phase_data_phase)
    return CubicSplineInterpolation(phase_range,phase_data_phase)
end

function phase_cos_blox(ω,t,phase_inter::F) where F
    if (t in phase_inter.itp.ranges[1])
        phase = phase_inter(t)
    else
        phase = 0.0
    end
    return cos(ω*t + phase)
end

function phase_sin_blox(ω,t,phase_inter::F) where F
    if (t in phase_inter.itp.ranges[1])
        phase = phase_inter(t)
    else
        phase = 0.0
    end
    return sin(ω*t + phase)
end
