## Load Packages
using Neuroblox, MAT, ModelingToolkit, OrdinaryDiffEq, DiffEqCallbacks, Plots

## Load Experimental Data
fs         = 1000
## Cortical
# P1
data       = matread("examples/learningmodels/phi_p1.mat")
pdata_P1   = data["phi_p1"][1:5000]
prange_P1  = 0:(1/fs):(length(pdata_P1)-1)*(1/fs)
dataset_P1 = [prange_P1, pdata_P1]
# P2
data       = matread("examples/learningmodels/phi_p2.mat")
pdata_P2   = data["phi_p2"][1:5000]
prange_P2  = 0:(1/fs):(length(pdata_P2)-1)*(1/fs)
dataset_P2 = [prange_P2, pdata_P2]
## Subcortical
# S1
data       = matread("examples/learningmodels/phi_s1.mat")
pdata_S1   = data["phi_s1"][1:5000]      
prange_S1  = 0:(1/fs):(length(pdata_S1)-1)*(1/fs)
dataset_S1 = [prange_S1, pdata_S1]
# S2
data       = matread("examples/learningmodels/phi_s2.mat")
pdata_S2   = data["phi_s2"][1:5000]       
prange_S2  = 0:(1/fs):(length(pdata_S2)-1)*(1/fs)
dataset_S2 = [prange_S2, pdata_S2]

## Declare Parameters (
# Cortical
ω_P = 20*(2*pi)
d_P = 30
# Subcortical
ω_S = 20*(2*pi)
d_S = 30

## Create Circuit with 4 Loops
# P1<->S1
@named P1S1 = create_rl_loop(
    ROIs       = ["P1", "S1"],
    parameters = Dict(:ω => (ω_P, ω_S), :d => (d_P, d_S)),
    datasets   = [dataset_P1, dataset_S1],
    c_ext      = 0.04
    )
# P2<->S2
@named P2S2 = create_rl_loop(
    ROIs       = ["P2", "S2"],
    parameters = Dict(:ω => (ω_P, ω_S), :d => (d_P, d_S)),
    datasets   = [dataset_P2, dataset_S2],
    c_ext      = 0.04
    )
# P2<->S1
@named P2S1 = create_rl_loop(
    ROIs       = ["P2", "S1"],
    parameters = Dict(:ω => (ω_P, ω_S), :d => (d_P, d_S)),
    datasets   = [dataset_P2, dataset_S1],
    c_ext      = 0.04
    )
# P1<->S2
@named P1S2 = create_rl_loop(
    ROIs       = ["P1", "S2"],
    parameters = Dict(:ω => (ω_P, ω_S), :d => (d_P, d_S)),
    datasets   = [dataset_P1, dataset_S2],
    c_ext      = 0.04
    )

## Compose Circuit Model
corticostriatal_circuit = compose(P1S1, P2S2, P2S1, P1S2)

## Solve Model
prob = ODEProblem(
    structural_simplify(corticostriatal_circuit),  [], (0, sim_dur), []
    )
sol  = solve(prob, Rodas3(), saveat=1/fs, dt=1/fs) 

## Plot Solution
plot(sol.t,  sol[1,:], label="NeuralMassP1.x", lw=:1.6, lc=:green)
plot!(sol.t, sol[3,:], label="NeuralMassS1.x", lw=:1.6, lc=:purple)
plot!(sol.t, sol[5,:], label="NeuralMassP2.x", lw=:1.6, lc=:lime)
plot!(sol.t, sol[7,:], label="NeuralMassS2.x", lw=:1.6, lc=:mediumpurple)
xlabel!("seconds")
ylabel!("amplitude")
title!("Simulated LFP")
xlims!(1,2)
xticks!(0:0.5:sim_dur)