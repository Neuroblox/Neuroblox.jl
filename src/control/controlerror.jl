"""
ARVTarget
Time series data is bandpass filtered and then the power spectrum
is computed for a given time interval (control bin), returned as
the average value of the power spectral density within a certain
frequency band ([lb, ub]).
"""
function ARVTarget(data, lb, ub, fs, order, control_bin)
    signal = Neuroblox.bandpassfilter(data=data, lb=lb, ub=ub, fs=fs, order=order)
    periodogram_estimation = periodogram(vec(signal), fs=fs, window=hanning)
    pxx = periodogram_estimation.power
    f = periodogram_estimation.freq
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
Circular difference is quantified as the angle of circular_location.
"""
function CDVTarget(data, lb, ub, fs, order)
    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end
    signal = Neuroblox.bandpassfilter(data=data, lb=lb, ub=ub, fs=fs, order=order)
    phi = Neuroblox.phaseangle(data=signal)
    circular_location = exp.(im*phi)
    return circular_location
end

"""
PDVTarget
Time series data is bandpass filtered and hilbert-transformed.
Phase angle is computed in radians.
Phase deviation is quantified as the angle difference between a given set of signals.
"""
function PDVTarget(data, lb, ub, fs, order)
    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end
    signal = Neuroblox.bandpassfilter(data=data, lb=lb, ub=ub, fs=fs, order=order)
    phi = Neuroblox.phaseangle(data=signal)
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
    signal1 = Neuroblox.bandpassfilter(data=data1, lb=lb, ub=ub, fs=fs, order=order)
    signal2 = Neuroblox.bandpassfilter(data=data2, ls=lb, ub=ub, fs=fs, order=order)
    dphi    = Neuroblox.phaseangle(data=signal1) .- Neuroblox.phaseangle(data=signal2)
    PLV     = abs(mean(exp.(im*dphi)))
    return PLV
end

function ACVTarget(data1, data2, lb, ub, fs, order)
    if typeof(data1) == Matrix{Float64}
        data1 = vec(data1)
    end
    if typeof(data2) == Matrix{Float64}
        data2 = vec(data2)
    end
    avg_coh(x)  = dropdims(mean(coherence(x); dims=3); dims=3)
    signal1 = Neuroblox.bandpassfilter(data=data1, lb=lb, ub=ub, fs=fs, order=order)
    signal2 = Neuroblox.bandpassfilter(data=data2, lb=lb, ub=ub, fs=fs, order=order)
    signals = Matrix{Float64}(undef, 2, length(signal1))
    signals[1,:] = signal1
    signals[2,:] = signal2
    avgcoh = avg_coh(mt_coherence(signals; demean=true, fs=fs, freq_range=(lb,ub)))
    ACV = avgcoh[2,1]
    return ACV
end

"""
ControlError
Returns the control error (deviation of the actual value from the target value).
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

    if type == "ACV"
        control_error = ACVTarget(target, actual, lb, ub, fs, order)
    end

    return control_error
end