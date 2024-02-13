using DifferentialEquations, Random, Distributions, DataFrames, CSV

# Neuron network definition

# This works by organizing 2 states per neuron: voltage and synaptic current
# You need to specify the number of neurons and the adjacency matrix

function qif_mpr_network!(du, u, p, t)
    η, vₚ, vᵣ, sⱼ, J, τₛ, N, adj_matr, Iₑ = p
    
    du[1:2:end] .= u[1:2:end] .^ 2 .+ η .+ Iₑ .+ J .* (u[2:2:end])
    du[2:2:end] .= -u[2:2:end] ./ τₛ
end

function qif_mpr_threshold(u, t, integrator)
    return sum(integrator.u[1:2:end] .>= integrator.p[2]) > 0
end

function reset!(integrator)
    idx = findall(x -> x >= integrator.p[2], integrator.u[1:2:end])
    integrator.u[2 .*idx .- 1] .= integrator.p[3]

    spikes = zeros(integrator.p[7], 1)
    spikes[idx] .= 1
    integrator.u[2:2:end] .+= integrator.p[4]*integrator.p[8]*spikes/integrator.p[7]
end

# Opting for discrete callbacks rather than continuous callbacks for simplicity
threshold = DiscreteCallback(qif_mpr_threshold, reset!)

# Kept a preset time callback around so I don't have to go looking for it later. It doesn't do anything meaningful at the moment.
# It will eventually add a current at a given time.
current_step = PresetTimeCallback(10, integrator -> integrator.p[9] += 3.0)
current_stop = PresetTimeCallback(40, integrator -> integrator.p[9] -= 3.0)
cb = CallbackSet(current_step, current_stop, threshold)

# Hyperparameters
N =10000

# Set the shape of the Lorentzian for quenched noise
μ = -5
σ = 1

# Parameters
η = rand(Cauchy(μ, σ), N, 1)
vₚ = 200
vᵣ = -200
sⱼ = 1.0 # Synaptic increase for each spike - holdover from CC paper
J = 15.0
τₛ = 1.0 # Not specified by MPR?
adj_matr = ones(N, N)
Iₑ = 0.0 # Can also be incremented by the PresetTimeCallback

p = [η, vₚ, vᵣ, sⱼ, J, τₛ, N, adj_matr, Iₑ]
u₀ = zeros(2*N, 1)
tspan = (0.0, 50.0)

prob = ODEProblem(qif_mpr_network!, u₀, tspan, p, callback=cb)
# Solver performance notes:
# LSODA is not compatible with callbacks
# CVODE_BDF actually doesn't seem to provide a speed increase
# Tsit5 seems to do fine and is decently fast
# not specifying the solver results in Tsit5
# Vern7 doesn't work (presumably too stiff?)
@time sol = solve(prob, Tsit5(), saveat=0.5)
df = DataFrame(sol)
CSV.write("/Users/achesebro/Downloads/test_qif_1e4N_diffeq.csv", df)