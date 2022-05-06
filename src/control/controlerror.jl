"""
ARVTarget
Time series data is bandpass filtered and then the power spectrum
is computed for a given time interval (control bin), returned as
the average value of the power spectral density within a certain
frequency band ([lb, ub]).
"""
function ARVTarget(data, lb, ub, fs, control_bin)
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, 4)
    f, pxx = Neuroblox.powerspectrum(signal, control_bin, fs, "periodogram", hanning)
    lbs = Int(ceil(lb*length(f)/500))
    ubs = Int(ceil(ub*length(f)/500))
    value = abs.(pxx)[lbs:ubs]
    arv = Statistics.mean(value)
    return arv
end

"""
PhaseTarget
Time series data is bandpass filtered and hilbert-transformed.
Phase angle is computed in radians.
"""
function PhaseTarget(data, lb, ub, fs)
    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, 4)
    phi = Neuroblox.phaseangle(signal)
    circular_location = exp.(im*phi)
    return circular_location
end

"""
ControlError
Returns the control error (deviation of the actual signal from the target signal) measured according
to some type (ARV value, relies on ARVTarget, or phase value, relies on PhaseTarget).
"""
function ControlError(type, target, actual, lb, ub, fs, call_rate)

    control_bin = call_rate*fs
    if type == ARV
        arv_target = Neuroblox.ARVTarget(target, lb, ub, fs, control_bin)
        arv_actual = Neuroblox.ARVTarget(actual, lb, ub, fs, control_bin)
        control_error = arv_target - arv_actual
    end

    if type == phase
        phi_target = Neuroblox.PhaseTarget(target, lb, ub, fs)
        phi_actual = Neuroblox.PhaseTarget(actual, lb, ub, fs)
        control_error = angle.(phi_target./phi_actual)
    end

    return control_error
end