"""
SpectralTools.jl

This utility file contains the methods utilized for spectral data analysis.
    PowerSpectrum : compute power spectrum from time series
    mar2csd       : compute cross-spectral densities from multivariate auto-regressive model parameters
    csd2mar       : compute multivariate auto-regressive model parameters from cross-spectral densities
"""

function PowerSpectrum(;name, data=data, T=T, uniform=true, dt=dt, NQ=500)
    
    """
    PowerSpectrum computes the power spectrum of a given time series signal. 
    It has the following inputs:
        'data'   : time series data which assumes time is in the first column of the data matrix
        'T'      : time series signal duration (in seconds)
        'uniform': argument to account for uniform (uniform=true) or nonuniform(uniform=false) sampling
        'fs'     : sampling frequency
        'NQ'     : nyquist frequency (set to a default value for nonuniform sampling)
    The following outputs:
        'f'      : frequency vector with sampling df (frequency resolution) 
        'pxx'    : real part of the power spectrum estimate
    With parameters:
        'df'     : frequency resolution
    
    The time series signal is resampled according to the frequency resolution to account for non-uniform sampling.
    
    """
        
    if uniform == true
        fs = 1/dt 
        NQ = fs/2            
    else
        fs = 1/(NQ*2)                                  
    end

    df = 1/T                                           
    f = 0:df:100                  
    pxx = df*(2*(dt^2))*AbstractFFTs.fft(DSP.resample(data, length(f))).*conj(AbstractFFTs.fft(DSP.resample(data, length(f))))
    pxx = real(pxx)

    return f, pxx
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
