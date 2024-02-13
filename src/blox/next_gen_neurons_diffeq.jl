using DifferentialEquations, Random, Distributions, DataFrames, CSV
using Plots

function izh!(du, u, p, t)
    a, b, c, d, I = p

    du[1] = 0.04*u[1]^2 + 5*u[1] + 140 - u[2] + I
    du[2] = a*(b*u[1] - u[2])
end

function thr(u, t, integrator)
    integrator.u[1] >= 30
end

function reset!(integrator)
    integrator.u[1] = integrator.p[3]
    integrator.u[2] += integrator.p[4]
end

threshold = DiscreteCallback(thr, reset!)
current_step = PresetTimeCallback(50, integrator -> integrator.p[5] += 10)
cb = CallbackSet(current_step, threshold)

p = [0.02, 0.2, -50, 2, 0]
u0 = [-65, p[2]*-65]
tspan = (0.0, 300.0)

prob = ODEProblem(izh!, u0, tspan, p, callback=cb)
sol = solve(prob, Tsit5())
plot(sol)

# Neuron network definition

# This works by organizing 3 states per neuron: voltage, recovery, and synaptic current
# You need to specify the number of neurons and the adjacency matrix

function izh_cc_network!(du, u, p, t)
    α, η, a, b, vₚ, vᵣ, wⱼ, sⱼ, eᵣ, gₛ, τₛ, N, adj_matr, Iₑ = p
    
    du[1:3:end] .= u[1:3:end] .* (u[1:3:end] .- α) .- u[2:3:end] .+ η .+ Iₑ .+ gₛ .* u[3:3:end] .* (eᵣ .- u[1:3:end])
    du[2:3:end] .= a .* (b .* u[1:3:end] .- u[2:3:end])
    du[3:3:end] .= -u[3:3:end] ./ τₛ
end

function izh_cc_threshold(u, t, integrator)
    return sum(integrator.u[1:3:end] .>= integrator.p[5]) > 0
end

function reset!(integrator)
    idx = findall(x -> x >= integrator.p[5], integrator.u[1:3:end])
    integrator.u[3 .*idx .- 2] .= integrator.p[6]
    integrator.u[3 .*idx .- 1] .+= integrator.p[7]

    spikes = zeros(integrator.p[12], 1)
    spikes[idx] .= 1
    integrator.u[3:3:end] .+= integrator.p[8]*integrator.p[13]*spikes/integrator.p[12]
end

# Opting for discrete callbacks rather than continuous callbacks for simplicity
threshold = DiscreteCallback(izh_cc_threshold, reset!)

# Kept a preset time callback around so I don't have to go looking for it later. It doesn't do anything meaningful at the moment.
# It will eventually add a current at a given time.
current_step = PresetTimeCallback(50, integrator -> integrator.p[14] += 0)
cb = CallbackSet(current_step, threshold)

# Hyperparameters
N =2000

# Set the shape of the Lorentzian for quenched noise
μ = 0.12
σ = 0.02

# Parameters
α = 0.6215
η = rand(Cauchy(μ, σ), N, 1)
a = 0.0077
b = -0.0062
vₚ = 200
vᵣ = -200
wⱼ = 0.0189
sⱼ = 1.2308
eᵣ = 1.0
gₛ = 1.2308
τₛ = 2.6
adj_matr = ones(N, N)
Iₑ = 0.0 # Can also be incremented by the PresetTimeCallback

p = [α, η, a, b, vₚ, vᵣ, wⱼ, sⱼ, eᵣ, gₛ, τₛ, N, adj_matr, Iₑ]
u₀ = zeros(3*N, 1)
tspan = (0.0, 800.0)

prob = ODEProblem(izh_cc_network!, u₀, tspan, p, callback=cb)
# Solver performance notes:
# LSODA is not compatible with callbacks
# CVODE_BDF actually doesn't seem to provide a speed increase
# Tsit5 seems to do fine and is decently fast
# not specifying the solver results in Tsit5
# Vern7 doesn't work (presumably too stiff?)
@time sol = solve(prob, Tsit5(), saveat=2)
df = DataFrame(sol)
CSV.write("/Users/achesebro/Downloads/test_500N_diffeq.csv", df)