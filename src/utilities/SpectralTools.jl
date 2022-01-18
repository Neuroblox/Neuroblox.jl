"""
SpectralTools.jl

This utility file contains the methods utilized for spectral data analysis.
    PowerSpectrum

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