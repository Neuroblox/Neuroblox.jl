"""
spectral_tools.jl

This utility file contains the methods utilized for spectral data analysis.
    powerspectrum  : compute power spectrum from time series via autopower, periodogram, and pwelch periodogram
    complexwavelet : generate a complex morlet wavelet
    bandpassfilter : designs a bandpass filter and applies it to time series data with some given pass band
    mar2csd        : compute cross-spectral densities from multivariate auto-regressive model parameters
    csd2mar        : compute multivariate auto-regressive model parameters from cross-spectral densities
    mar_ml         : maximum likelihood estimate of multivariate auto-regressive model parameters
"""

"""
powerspectrum computes the power spectrum of a given time series signal. 
Data in matrix format is converted to vector format.
It has the following inputs:
    'data'   : time series data which assumes time is in the first column of the data matrix
    'T'      : time series signal duration (in seconds)
    'fs'     : sampling frequency (default=1000)
    'method' : select from auto, periodogram, pwelch (default=pwelch)
    'window' : select from none, hanning, or hamming (default=hanning)
The following outputs:
    'f'      : frequency vector with sampling df (frequency resolution) 
    'pxx'    : real part of the power spectrum estimate
With parameters:
    'df'     : frequency resolution
"""
function powerspectrum(data, T, fs, method, window)

    if typeof(data) == Matrix{Float64}
        data = vec(data)
    end

    if method == "auto" 
        df = 1/T                                           
        f = 0:df:(fs/2)  
        dt = 1/fs             
        pxx = df*(2*(dt^2))*AbstractFFTs.fft(DSP.resample(data, length(f))).*conj(AbstractFFTs.fft(DSP.resample(data, length(f))))
        pxx = real(pxx)
    end

    if method == "periodogram"
        periodogram_estimation = periodogram(data, fs=fs, window=window)
        pxx = periodogram_estimation.power
        f = periodogram_estimation.freq
    end

    if method == "pwelch"
        pwelch_periodogram_estimation = welch_pgram(data, fs=fs, window=window)
        pxx = pwelch_periodogram_estimation.power
        f = pwelch_periodogram_estimation.freq
    end

    return f, pxx
end

"""
This function takes in time series data and bandpass filters it.
It has the following inputs:
    data: time series data
    lb: minimum cut-off frequency
    ub: maximum cut-off frequency
    fs: sampling frequency
    order: filter order
"""
function bandpassfilter(data, lb, ub, fs, order)
    responsetype = Bandpass(lb, ub, fs=fs)
    order = Int(ceil(0.5*order))
    designmethod = Butterworth(order)
    signal = filtfilt(digitalfilter(responsetype, designmethod), data)
    return signal
end

"""
This function takes in time series data and hilbert transforms it using the DSP hilbert function.
"""
function hilberttransform(data)
    transformed_data = DSP.hilbert(data)
    return transformed_data
end

"""
This function creates a complex morlet wavelet by windowing a complex sine wave with a Gaussian taper. 
The morlet wavelet is a special case of a bandpass filter in which the frequency response is Gaussian-shaped.
Convolution with a complex wavelet is equivalent to performing a Hilbert transform of a bandpass filtered signal.

It has the following inputs:
    data: time series data 
    dt  : data sampling rate 
    lb  : lower bound wavelet frequency (in Hz)
    ub  : upper bound wavelet frequency (in Hz)
    a   : amplitude of the Gaussian taper, default is 1
    n   : number of wavelet cycles of the Gaussian taper, defines the trade-off between temporal precision and frequency precision
          larger n gives better frequency precision at the cost of temporal precision
          default is 6 Hz
    m   : x-axis offset, default is 0
    num_wavelets : number of wavelets to create, default is 5

And outputs:
    complex_wavelet : a family of complex morlet wavelets
"""
function complexwavelet(data, dt, lb, ub, a=1, n=6, m=0, num_wavelets=5)

    fs = 1/dt
    t = -length(data)/2:1/fs:length(data)/2
    f = LinRange(lb, ub, num_wavelets) 

    complex_wavelets = []
    for i = 1:num_wavelets

        # Create Gaussian Taper
        s = n/2*π*f[i] 
        gauss_window = a*exp.((-(t.-m).^2)/(2*(s.^2)))
        
        # Create Complex Sine Function
        A = 1/((s*sqrt(π)).^0.5)
        complex_sine = A.*exp.(im*2*π*f[i]*t)

        # Create Kernel
        wavelet_temp = complex_sine.*gauss_window
        push!(complex_wavelets, wavelet_temp)
    end

    return complex_wavelets

end


"""
This function converts multivariate auto-regression (MAR) model parameters to a cross-spectral density (CSD).
A     : coefficients of MAR model, array of length p, each element contains the regression coefficients for that particular time-lag.
Σ     : noise covariance matrix of MAR
p     : number of time lags
freqs : frequencies at which to evaluate the CSD
sf    : sampling frequency

This function returns:
csd   : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
"""
function mar2csd(mar, freqs, sf)
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2*pi*freqs/sf
    nf = length(w)
	csd = zeros(ComplexF64, nf, nd, nd)
	for i = 1:nf
		af_tmp = la.I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'
	end
    csd = 2*csd/sf
    return csd
end

function mar2csd(mar, freqs)
    sf = 2*freqs[end]   # freqs[end] is not the sampling frequency of the signal... not sure about this step.
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2pi*freqs/sf
    nf = length(w)
	csd = zeros(ComplexF64, nf, nd, nd)
	for i = 1:nf
		af_tmp = la.I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'
	end
    csd = 2*csd/sf
    return csd
end


"""
This function converts a cross-spectral density (CSD) into a multivariate auto-regression (MAR) model. It first transforms the CSD into its
cross-correlation function (Wiener-Kinchine theorem) and then computes the MAR model coefficients.
csd       : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
w         : frequencies
dt        : time step size
p         : number of time steps of auto-regressive model

This function returns
coeff     : array of length p of coefficient matrices of size sqrt(N)xsqrt(N)
noise_cov : noise covariance matrix
"""
function csd2mar(csd, w, dt, p)
    # TODO: investiagate why SymmetricToeplitz(ccf[1:p, i, j]) is not good to be used but instead need to use Toeplitz(ccf[1:p, i, j], ccf[1:p, j, i])
    # as is done in the original MATLAB code (confront comment there). ccf should be a symmetric matrix so there should be no difference between the
    # Toeplitz matrices but for the second jacobian (J[2], see for loop for i = 1:nJ in function diff) the computation produces subtle differences between
    # the two versions.

    dw = w[2] - w[1]
    w = w/dw
    ns = dt^-1
    N = ceil(Int64, ns/2/dw)
    gj = findall(x -> x > 0 && x < (N + 1), w)
    gi = gj .+ (ceil(Int64, w[1]) - 1)    # TODO: figure out what's the purpose of this!
    g = zeros(ComplexF64, N)

    # transform to cross-correlation function
    ccf = zeros(ComplexF64, N*2+1, size(csd,2), size(csd,3))
    for i = 1:size(csd, 2)
        for j = 1:size(csd, 3)
            g[gi] = csd[gj,i,j]
            f = ifft(g)
            f = ifft(vcat([0.0im; g; conj(g[end:-1:1])]))
            ccf[:,i,j] = real.(fftshift(f))*N*dw
        end
    end

    # MAR coefficients
    N = size(ccf,1)
    m = size(ccf,2)
    n = (N - 1) ÷ 2
    p = min(p, n - 1)
    ccf = ccf[(1:n) .+ n,:,:]
    A = zeros(m*p, m)
    B = zeros(m*p, m*p)
    for i = 1:m
        for j = 1:m
            A[((i-1)*p+1):i*p, j] = ccf[(1:p) .+ 1, i, j]
            B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = tm.Toeplitz(ccf[1:p, i, j], vcat(ccf[1,i,j], ccf[2:p, j, i]))  # SymmetricToeplitz(ccf[1:p, i, j])
        end
    end
    a = B\A

    Σ  = ccf[1,:,:] - A'*a   # noise covariance matrix
    lags = [-a[i:p:m*p, :] for i = 1:p]
    mar = Dict([("A", lags), ("Σ", Σ), ("p", p)])
    return mar
end


"""
Maximum likelihood estimator of a multivariate, or vector auto-regressive model.
    y : MxN Data matrix where M is number of samples and N is number of dimensions
    p : time lag parameter, also called order of MAR model
    return values
    mar["A"] : model parameters is a NxNxP tensor, i.e. one NxN parameter matrix for each time bin k ∈ {1,...,p}
    mar["Σ"] : noise covariance matrix
"""
function mar_ml(y, p)
    (ns, nd) = size(y)
    ns < nd && error("error: there are more covariates than observation")
    y = transpose(y)
    Y = y[:, p+1:ns]
    X = zeros(nd*p, ns-p)
    for i = p:-1:1
        X[(p-i)*nd+1:(p-i+1)*nd, :] = y[:, i:ns+i-p-1]
    end

    A = (Y*X')/(X*X')
    ϵ = Y - A*X
    Σ = ϵ*ϵ'/ns   # unbiased estimator requires the following denominator (ns-p-p*nd-1), the current is consistent with SPM12
    A = -[A[:, (i-1)*nd+1:i*nd] for i = 1:p]    # flip sign to be consistent with SPM12 convention
    mar = Dict([("A", A), ("Σ", Σ), ("p", p)])
    return mar
end
