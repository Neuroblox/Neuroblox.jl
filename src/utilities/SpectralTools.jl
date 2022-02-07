"""
SpectralTools.jl

This utility file contains the methods utilized for spectral data analysis.
    powerspectrum  : compute power spectrum from time series via autopower, periodogram, and pwelch periodogram
    complexwavelet : generate a complex morlet wavelet
    mar2csd        : compute cross-spectral densities from multivariate auto-regressive model parameters
    csd2mar        : compute multivariate auto-regressive model parameters from cross-spectral densities
"""

"""
powerspectrum computes the power spectrum of a given time series signal. 
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
function powerspectrum(;name, data=data, T=T, fs=1000, method="pwelch", window="hanning")

    if method == "auto" 
        df = 1/T                                           
        f = 0:df:(fs/2)               
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
function complexwavelet(;name, data=data, dt=dt, lb=lb, ub=ub, a=1, n=6, m=0, num_wavelets=5)

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

function mar2csd(coeff, noise_cov, p, freqs)
    """
    This function converts multivariate auto-regression (MAR) model parameters to a cross-spectral density (CSD).
    coeff     : coefficients of MAR model, array of length p, each element contains the regression coefficients for that particular time-lag.
    noise_cov : noise covariance matrix of MAR
    p         : number of time lags
    freqs     : frequencies at which to evaluate the CSD

    This function returns:
    csd       : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
    """

    dim = size(noise_cov, 1)
    sf = 2*freqs[end]
    w  = 2*pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz?
    nf = length(w)
	csd = zeros(ComplexF64, nf, dim, dim)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + coeff[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * noise_cov * iaf_tmp'     # is this really the covariance or rather precision?!
	end
    csd = 2*csd/sf

    return csd
end


function csd2mar(csd, w, dt, p)
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
            B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = Toeplitz(ccf[1:p, i, j], vcat(ccf[1,i,j], ccf[2:p, j, i]))  # SymmetricToeplitz(ccf[1:p, i, j])
        end
    end
    a = B\A

    noise_cov  = ccf[1,:,:] - A'*a
    coeff = [-a[i:p:m*p, :] for i = 1:p]

    return (coeff, noise_cov)
end
