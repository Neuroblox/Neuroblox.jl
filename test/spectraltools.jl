using Neuroblox, Test

using Distributions
using LinearAlgebra: I, Matrix


nd = 2  # number of dimensions

p = 2  # number of time lags of MAR model
f = 2.0.^(range(0,5)) # frequencies at which to evaluate CSD
dt = 1/(2*f[end]) # time step, inverse of sampling frequency
dist = InverseWishart(nd*2, Matrix(1.0I, nd, nd))
Σ = rand(dist)   # noise covariance matrix of MAR model
a = [randn(nd, nd) for i = 1:p]   # MAR model parameters

csd = mar2csd(a, Σ, p, f)
a_est, Σ_est = csd2mar(csd, f, dt, p)

@test_broken a ≈ a_est
@test_broken Σ ≈ Σ_est