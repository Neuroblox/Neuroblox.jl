"""
ARVTarget
Time series data is bandpass filtered and then the power spectrum
is computed for a given time interval (control bin), returned as
the average value of the power spectral density within a certain
frequency band ([lb, ub]).
"""
function ARVTarget(data, lb, ub, fs, order, control_bin)
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, order)
    f, pxx = Neuroblox.powerspectrum(signal, control_bin, fs, "periodogram", hanning)
    lbs = Int(ceil(lb*length(f)/500))
    ubs = Int(ceil(ub*length(f)/500))
    value = abs.(pxx)[lbs:ubs]
    arv = Statistics.mean(value)
    return arv
end

"""
CDVTarget
Time series data is bandpass filtered and hilbert-transformed.
Phase angle is computed in radians.
"""
function CDVTarget(data, lb, ub, fs, order)
    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, order)
    phi = Neuroblox.phaseangle(signal)
    circular_location = exp.(im*phi)
    return circular_location
end

"""
PDVTarget
Time series data is bandpass filtered and hilbert-transformed.
Phase angle is computed in radians.
"""
function PDVTarget(data, lb, ub, fs, order)
    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, order)
    phi = Neuroblox.phaseangle(signal)
    return phi
end

"""
PLVTarget
Time series data is bandpass filtered and hilbert-transformed.
Phase angle is computed in radians.
"""
function PLVTarget(data1, data2, lb, ub, fs, order)
    if typeof(data1) == Matrix{Float64}
        data1 = vec(data1)
    end
    if typeof(data2) == Matrix{Float64}
        data2 = vec(data2)
    end
    signal1 = Neuroblox.bandpassfilter(data1, lb, ub, fs, order)
    signal2 = Neuroblox.bandpassfilter(data2, lb, ub, fs, order)
    dphi    = Neuroblox.phaseangle(signal1) .- Neuroblox.phaseangle(signal2)
    PLV     = abs(mean(exp.(im*dphi)))
    return PLV
end

"""
ControlError
Returns the control error (deviation of the actual signal from the target signal) measured according
to some type (ARV value, relies on ARVTarget, or phase value, relies on PhaseTarget).
"""
function ControlError(type, target, actual, lb, ub, fs, order, call_rate)

    control_bin = call_rate*fs

    if type == "ARV"
        arv_target = Neuroblox.ARVTarget(target, lb, ub, fs, order, control_bin)
        arv_actual = Neuroblox.ARVTarget(actual, lb, ub, fs, order, control_bin)
        control_error = arv_target - arv_actual
    end

    if type == "CDV"
        cdv_target = Neuroblox.CDVTarget(target, lb, ub, fs, order)
        cdv_actual = Neuroblox.CDVTarget(actual, lb, ub, fs, order)
        control_error = angle.(cdv_target./cdv_actual)
    end

    if type == "PDV"
        target = Neuroblox.PDVTarget(target, lb, ub, fs, order)
        actual = Neuroblox.PDVTarget(actual, lb, ub, fs, order)
        control_error = angle.(exp.(im*(abs.(target-actual))))
    end

    if type == "PLV"
        control_error = PLVTarget(target, actual, lb, ub, fs, order)        
    end

    return control_error
end