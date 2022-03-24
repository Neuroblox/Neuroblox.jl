using Neuroblox, Test

import LinearAlgebra as la
using Distributions
using OrdinaryDiffEq, Plots
using Statistics
using MAT
using DSP


nd = 2  # number of dimensions

p = 2  # number of time lags of MAR model
f = 2.0.^(range(0,stop=5)) # frequencies at which to evaluate CSD
dt = 1/(2*f[end]) # time step, inverse of sampling frequency
dist = InverseWishart(nd*2, la.Matrix(1.0la.I, nd, nd))
Σ = rand(dist)   # noise covariance matrix of MAR model
a = [randn(nd, nd) for i = 1:p]   # MAR model parameters
mar = Dict([("A", a), ("Σ", Σ), ("p", p)])

csd = mar2csd(mar, f)
a_est, Σ_est = csd2mar(csd, f, dt, p)

@test_broken a ≈ a_est
@test_broken Σ ≈ Σ_est


"""
Design Test Case (PowerSpectrum). 
A circuit model is used, with an explicit paramter setting the oscillation frequency range of the system in radians/second:
    ω = 4*2*π
The test will count the number of peaks in a given time window and match that to the frequency containing the most power.
"""
freq_of_interest = 4
# Create Circuit 
@named STN = harmonic_oscillator(ω=freq_of_interest*2*π, ζ=1, k=(freq_of_interest*2*π)^2, h=5.0)
sys = [STN]
adj_matrix = [1.0]
@named BG_Circuit = LinearConnections(sys=sys, adj_matrix=adj_matrix, connector = [s.x for s in sys])
# Run Simulation
sim_dur = 5.0                                                                           # Simulation time (seconds)
prob = ODAEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5(), dt = 0.001)

"""
powerspectrum test.
The power spectrum of the solution should have a center frequency around that set by ω in Hz, with some tolerance.
To Do: Employ a peak detection algorithm, useful for this test and signal processing. Number of peaks should match center frequency.
"""
f, pxx = Neuroblox.powerspectrum(sol[1,:], sim_dur, 1000, "auto", hanning)
# 1) Make sure you can plot the data
Plots.plot(sol.t, sol[1,:])
Plots.plot(f, pxx, xlims=(0,10))
# 2) Find the local maxima
T = sim_dur
df = 1/T
find_max = findmax(pxx[1:length(f)-Int(ceil(freq_of_interest/df))])
index_of_maximum = find_max[2]
tol = 0.5
@test index_of_maximum*df>4-tol
@test index_of_maximum*df<4+tol

"""
bandpassfilter test
Compute power spectrum before and after filtering the data.
"""
data = matread("lfp_test_data.mat")
data = data["lfp"]
f, pxx = Neuroblox.powerspectrum(data, length(data), 1000, "periodogram", hanning)

lb = 12
ub = 30
signal = Neuroblox.bandpassfilter(data, lb, ub, 1000, 4)
f_signal, pxx_signal = Neuroblox.powerspectrum(signal, length(data), 1000, "periodogram", hanning)

@test pxx_signal[1:lb] < pxx[1:lb]
@test pxx_signal[ub:100] < pxx[ub:100]

"""
complexwavelet test
Wavelets must have values near zero at both ends, as well as a mean value of zero
"""
data = matread("lfp_test_data.mat")
wavelets = Neuroblox.complexwavelet(data["lfp"], 0.001, 2, 60)
tol = 0.2
@test real(wavelets[1][1]) < tol
all_wavelets = Statistics.mean(real(wavelets))
average_over_all = sum(all_wavelets)/length(all_wavelets)
tol = 0.001
@test sum(average_over_all) < 0 + tol