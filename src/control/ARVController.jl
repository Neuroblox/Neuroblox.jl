"""
ARVController
The average rectified value of the power spectrum within
a desired frequency band is computed at each controller call.
ARVController uses Neuroblox functions bandpassfilter and powerspectrum,
and returns the arv_estimation for each control bin, as well as
the times at which the estimation was computed.
"""
function ARVController(data, fs, call_rate, lb, ub, filter_order)
    # Controller Parameters
    call_time = call_rate*fs
    window_num = Int(floor(length(data)/call_time))
    # Filtered Data
    signal = Neuroblox.bandpassfilter(data, lb, ub, fs, filter_order)
    # ARV Computation
    arv_estimation = []
    for win in 1:window_num
        call_start = Int(1 + win*call_time - call_time)
        call_stop  = Int(win*call_time)
        spectral_estimation = Neuroblox.powerspectrum(signal[call_start:call_stop], call_stop-call_start+1, fs, "periodogram", hanning)
        lbs = Int(ceil(lb*length(spectral_estimation[1])/500))
        ubs = Int(ceil(ub*length(spectral_estimation[1])/500))
        arv = abs.(spectral_estimation[2])[lbs:ubs]
        arv = Statistics.mean(arv)
        push!(arv_estimation, arv)
    end
    controller_call_times = (1:Int(floor(length(data)/call_time))).*(call_rate*fs)
    return controller_call_times, arv_estimation
end 