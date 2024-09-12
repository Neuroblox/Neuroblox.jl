"""
spectral_tools.jl

This utility file contains the methods utilized for spectral data analysis.
    powerspectrum    : compute power spectrum from time series via autopower, periodogram, and pwelch periodogram
    bandpassfilter   : designs a bandpass filter and applies it to time series data with some given pass band
    phaseangle       : computes the phase angle on hilbert-transformed time series data
    complexwavelet   : generate a complex morlet wavelet
    mar2csd          : compute cross-spectral densities from multivariate auto-regressive model parameters
    csd2mar          : compute multivariate auto-regressive model parameters from cross-spectral densities
    mar_ml           : maximum likelihood estimate of multivariate auto-regressive model parameters
"""

"""
    powerspectrum(data, duration, sampling_freq, method="periodogram", window=hanning)

Compute the power spectrum of a given time series signal.

# Arguments
- `data`: Time series data as a vector
- `duration`: Time series signal duration in seconds
- `sampling_freq`: Sampling frequency in Hz
- `method`: Computation method, either "periodogram" or "pwelch" (default: "periodogram")
- `window`: Window function to use, e.g., hanning, hamming (default: hanning)

# Returns
- `frequencies`: Vector of frequencies
- `power`: Vector of power spectral density estimates
"""
function powerspectrum(data::AbstractVector{T},
                       duration,
                       sampling_freq,
                       method="periodogram",
                       window=hanning) where T<:Number

    if method == "periodogram"
        spectrum = periodogram(data_vector, fs=sampling_freq, window=window)
        frequencies = spectrum.freq
        power = spectrum.power
    elseif method == "pwelch"
        spectrum = welch_pgram(data_vector, fs=sampling_freq, window=window)
        frequencies = spectrum.freq
        power = spectrum.power
    else
        throw(ArgumentError("Invalid method. Use 'periodogram' or 'pwelch'."))
    end
    
    return frequencies, power
end

"""
    get_powerspectrum(sol, states_names, target_states_names, target_population_type;
                      trim=0.0, sampling_rate=0.0, method="periodogram", window=hanning)

Compute the power spectrum for specific states in a SciML solution.

# Arguments
- `sol`: A SciML solution object
- `states_names`: Vector of state names as strings
- `target_states_names`: String specifying the target states
- `target_population_type`: String specifying the target population type
- `trim`: Time to trim from the beginning of the time series (in ms, default: 0.0)
- `sampling_rate`: Sampling rate for resampling if needed (in ms)
- `method`: Spectrum computation method ("periodogram" or "pwelch", default: "periodogram")
- `window`: Window function to use in spectrum computation (default: hanning)

# Returns
- `frequencies`: Vector of frequencies
- `power`: Vector of power spectral density estimates

# Example
```julia
@named WC1 = WilsonCowan()
@named WC2 = WilsonCowan()
@named JR = JansenRit()
g = MetaDiGraph()
add_blox!.(Ref(g), [WC1, WC2, JR])
adj = [-1 6 1; 6 -1 1]
create_adjacency_edges!(g, adj)
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 500), [])
sol = solve(prob, Rodas4())
states_names = string.(unknowns(sys))
target_states_names = "E(t)"
target_population_type = "WC"
f, pxx = get_powerspectrum(sol,
                           states_names,
                           target_states_names,
                           target_population_type;
                           trim=200,
                           sampling_rate=0.1)
```
"""
function get_powerspectrum(sol::SciMLBase.AbstractSolution,
                           states_names::Vector{String},
                           target_states_names::String,
                           target_population_type::String;
                           trim=0.0,
                           sampling_rate=0.0,
                           method="periodogram",
                           window=hanning)

    inds_target = get_inds(states_names, target_states_names)
    ind_population = get_inds(states_names, target_population_type)
    inds = intersect(inds_target, ind_population)

    t_raw = unique(sol.t)

    if !isapprox(std(diff(t_raw)), 0, atol=1e-10*(t_raw[2]-t_raw[1]))
        sampling_rate == 0.0 && throw(ArgumentError("The solution was not saved at a fixed time step. Please provide a 'sampling_rate' in milliseconds."))
        t_sampled = t_raw[1]:sampling_rate:t_raw[end]
    else
        t_sampled = t_raw
        sampling_rate = t_sampled[2] - t_sampled[1]
    end

    target = sol(t_sampled, idxs = inds)
    start = Int(round(trim/sampling_rate)) + 1
    target = Array(target)[:, start:end]
    target_mean = vec(mean(target, dims=1))

    sampling_freq = 1000/sampling_rate  # Convert to Hz
    duration = length(target_mean) / sampling_freq

    frequencies, power = powerspectrum(target_mean, duration, sampling_freq, method, window)
    return frequencies, power
end

get_inds(x::Vector{String}, name) = findall(x -> contains(x, name), x)

get_inds(x::Vector{SymbolicUtils.BasicSymbolic{Real}} , name) = findall(x -> contains(string(x), name), x)


mutable struct PowerSpectrumBlox <: SpectralUtilities
    T::Float64
    fs::Float64
    method::String
    window::Union{Function,AbstractVector,Nothing}
    PSDfunc::Function
    function PowerSpectrumBlox(;name, T=20, fs=1000, method="periodogram", window="hanning")
        new(T, fs, method, window, powerspectrum)
    end
end

"""
bandpassfilter takes in time series data and bandpass filters it.
It has the following inputs:
    data: time series data
    lb: minimum cut-off frequency
    ub: maximum cut-off frequency
    fs: sampling frequency
    order: filter order
"""
function bandpassfilter(;data, lb=0.0, ub=1000.0, fs=1000.0, order=4)
    # return unfiltered data when parameters are bad
    if (lb >= ub) | (ub > fs / 2)
        return data
    end
    responsetype = Bandpass(lb, ub, fs=fs)
    order = Int(ceil(0.5*order))
    designmethod = Butterworth(order)
    signal = filtfilt(digitalfilter(responsetype, designmethod), data)
    return signal
end

mutable struct BandPassFilterBlox <: SpectralUtilities
    name::String
    lb::Float64
    ub::Float64
    fs::Float64
    order::Int64
    util_func::Function
    function BandPassFilterBlox(;name,lb=0.0,ub=1000.0, fs=1000, order=4, util_func=bandpassfilter)
        new(string(name),lb, ub, fs, order, util_func)
    end
end

"""
phaseangle takes in time series data, hilbert transforms it, and estimates the phase angle.
"""
function phaseangle(;data)
    d     = DSP.hilbert(data)
    phase = angle.(d)
    return phase
end

mutable struct PhaseAngleBlox <: SpectralUtilities
    name::String
    util_func::Function
    function PhaseAngleBlox(;name, util_func=phaseangle)
        new(string(name),util_func)
    end
end

"""
complexwavelet creates a complex morlet wavelet by windowing a complex sine wave with a Gaussian taper. 
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
		af_tmp = I
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
	csd = zeros(eltype(Σ), nf, nd, nd)
	for i = 1:nf
		af_tmp = I
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
Plain implementation of idft because AD dispatch versions for ifft don't work still!
"""
function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{eltype(x)}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
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
    g = zeros(eltype(csd), N)

    # transform to cross-correlation function
    ccf = zeros(eltype(csd), N*2+1, size(csd,2), size(csd,3))
    for i = 1:size(csd, 2)
        for j = 1:size(csd, 3)
            g[gi] = csd[gj,i,j]
            f = idft(g)
            f = idft(vcat([0.0im; g; conj(g[end:-1:1])]))
            ccf[:,i,j] = real.(DSP.fftshift(f))*N*dw
        end
    end

    # MAR coefficients
    N = size(ccf,1)
    m = size(ccf,2)
    n = (N - 1) ÷ 2
    p = min(p, n - 1)
    ccf = ccf[(1:n) .+ n,:,:]
    A = zeros(eltype(csd), m*p, m)
    B = zeros(eltype(csd), m*p, m*p)
    for i = 1:m
        for j = 1:m
            A[((i-1)*p+1):i*p, j] = ccf[(1:p) .+ 1, i, j]
            B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = Toeplitz(ccf[1:p, i, j], vcat(ccf[1,i,j], ccf[2:p, j, i]))  # SymmetricToeplitz(ccf[1:p, i, j])
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
