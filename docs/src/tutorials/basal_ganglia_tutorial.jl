# # Basal Ganglia Model and Parkinson's Disease Simulation

# ## Introduction
# This tutorial demonstrates how to build and simulate a basal ganglia model using Neuroblox, based on the work of [Adam et al. (2022)](https://doi.org/10.1073/pnas.2120808119). We'll explore the model's behavior in both normal and Parkinsonian conditions, showcasing the emergence of pathological beta oscillations characteristic of Parkinson's disease.

# ![Full basal ganglia model in baseline condition](../assets/basal_ganglia_baseline.jpg)


# We'll start with simple components and gradually build up to the full basal ganglia circuit, demonstrating how to analyze the results at each stage.

using Neuroblox
using DifferentialEquations ## For building and solving differential equations problems
using MetaGraphs ## use its MetaGraph type to build the circuit
using CairoMakie ## For plotting
using Random ## For setting a random seed

Random.seed!(123) ## Set a random seed for reproducibility

# ## Isolated MSN network in baseline condition
# We'll start by simulating an isolated network of Medium Spiny Neurons (MSNs)

# Blox definition
@named msn = Striatum_MSN_Adam()
sys = get_system(msn)

## Check the system's variables (100 neurons, each with associated currents)
unknowns(sys)


# Create and solve the SDE problem
## Define simulation parameters
tspan = (0.0, 5500.0) ## Simulation time span [ms]
dt = 0.05 ## Time step for solving and saving [ms]

## Create a stochastic differential equation problem and use the RKMil method to solve it
prob = SDEProblem(sys, [], tspan, [])
sol = solve(prob, RKMil(); dt=dt, saveat=dt)

# Plot voltage of a single neuron
plot(sol, idxs=1, axis = (xlabel = "time (ms)", ylabel = "membrane potential (mV)"))

# Plot mean field
meanfield(msn, sol, axis = (xlabel = "time (ms)", ylabel = "membrane potential (mV)", title = "Mean Field"))

# Detect spikes and compute firing rate
spikes = detect_spikes(msn, sol; threshold=-55)
t, fr = mean_firing_rate(spikes, sol)

# Create a raster plot
rasterplot(msn, sol, threshold = -55.0, axis = (; title = "Neuron's Spikes - Mean Firing Rate: $(round(fr[1], digits=2)) spikes/s"))


# Compute and plot the power spectrum of the GABAa current
fig = Figure(size = (1500, 600))

powerspectrumplot(fig[1,1], msn, sol; state = "G",
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 5),
                        beta_label_position = (22, 5),
                        gamma_label_position = (60, 5),
                        axis = (; title = "FFT with no window"))

powerspectrumplot(fig[1,2], msn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_start = 5,
                        alpha_label_position = (8.5, 5),
                        beta_label_position = (22, 5),
                        gamma_label_position = (60, 5),
                        axis = (; title = "Welch's method + Hanning window"))
fig

# We can also run multiple simulations in parallel and compute the average power spectrum
ens_prob = EnsembleProblem(prob)
ens_sol = solve(ens_prob, RKMil(); dt=0.05, saveat=0.05, trajectories=5)

powerspectrumplot(msn, ens_sol; state = "G",
                  method=welch_pgram, window=hanning,
                  ylims=(1e-5, 10),
                  alpha_start = 8,
                  alpha_label_position = (8.5, 4),
                  beta_label_position = (22, 4),
                  gamma_label_position = (60, 4),
                  axis = (; title = "Welch's method + Hanning window + Ensemble"))
fig

# ## Core striatal network: MSN + FSI
# Now we'll add Fast-Spiking Interneurons (FSIs) to our model

global_ns = :g ## global name for the circuit. All components should be inside this namespace.
@named msn = Striatum_MSN_Adam(namespace=global_ns)
@named fsi = Striatum_FSI_Adam(namespace=global_ns)

assembly = [msn, fsi]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g, 2, 1, Dict(:weight=> 0.6/7.5, :density=>0.15))

# Connection parameters:
# - `density`: 0.15 means 15% of FSI neurons (population 2) connect to MSNs (population 1).
# - `weight`: Represents connection strength, calculated as:
#   maximal conductance (0.6) / (number of presynaptic neurons * density)

@named sys = system_from_graph(g)
prob = SDEProblem(sys, [], tspan, [])
ens_prob = EnsembleProblem(prob)
ens_sol = solve(ens_prob, RKMil(); dt=dt, saveat=dt, trajectories=5)

# Detect spikes and compute firing rates
spikes_msn = detect_spikes(msn, ens_sol[1]; threshold=-35)
t, fr_msn = mean_firing_rate(spikes_msn, ens_sol[1])

spikes_fsi = detect_spikes(fsi, ens_sol[1]; threshold=-25)
t, fr_fsi = mean_firing_rate(spikes_fsi, ens_sol[1])

# Let's see their raster plots and power spectra
fig = Figure(size = (1000, 800))
rasterplot(fig[1,1], msn, ens_sol[1], threshold = -35.0, axis = (; title = "MSN - Mean Firing Rate: $(round(fr_msn[1], digits=2)) spikes/s"))
rasterplot(fig[1,2], fsi, ens_sol[1], threshold = -35.0, axis = (; title = "FSI - Mean Firing Rate: $(round(fr_fsi[1], digits=2)) spikes/s"))

powerspectrumplot(fig[2,1], msn, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        xlims=(8, 100),
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3))

powerspectrumplot(fig[2,2], fsi, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        xlims=(8, 100),
                        alpha_label_position = (8.5, 3),
                        beta_label_position = (22, 3),
                        gamma_label_position = (60, 3))
fig

# FSIs exhibit a peak in gamma frequencies. Their inhibition onto MSNs suppresses the low beta-band activity seen in isolated MSN populations, without reducing MSN firing rates. This spectral shift reflects a change in MSN spiking dynamics under FSI influence, rather than a decrease in overall activity.

# ## Full basal ganglia model in baseline condition
# Now we'll add the GPe and STN to complete the full basal ganglia model

@named gpe = GPe_Adam(namespace=global_ns)
@named stn = STN_Adam(namespace=global_ns)

assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g, 1, 3, Dict(:weight => 2.5/33, :density => 0.33))
add_edge!(g, 2, 1, Dict(:weight => 0.6/7.5, :density => 0.15))
add_edge!(g, 3, 4, Dict(:weight => 0.3/4, :density => 0.05))
add_edge!(g, 4, 2, Dict(:weight => 0.165/4, :density => 0.1))

@named sys = system_from_graph(g)
prob = SDEProblem(sys, [], tspan, [])
ens_prob = EnsembleProblem(prob)
ens_sol = solve(ens_prob, RKMil(); dt=dt, saveat=dt, trajectories=5)

# Compute and plot power spectra for all components
fig = Figure(size = (1500, 600))
fig = Figure(size = (1500, 600))
powerspectrumplot(fig[1,1], msn, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_label_position = (8.5, 2),
                        beta_label_position = (22, 2),
                        gamma_label_position = (60, 2),
                        axis = (; title = "MSN"))

powerspectrumplot(fig[1,2], fsi, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_label_position = (8.5, 2),
                        beta_label_position = (22, 2),
                        gamma_label_position = (60, 2),
                        axis = (; title = "FSI"))

powerspectrumplot(fig[1,3], gpe, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_label_position = (8.5, 2),
                        beta_label_position = (22, 2),
                        gamma_label_position = (60, 2),
                        axis = (; title = "GPe"))

powerspectrumplot(fig[1,4], stn, ens_sol; state = "G",
                        method=welch_pgram, window=hanning,
                        ylims=(1e-5, 10),
                        alpha_label_position = (8.5, 2),
                        beta_label_position = (22, 2),
                        gamma_label_position = (60, 2),
                        axis = (; title = "STN"))

fig

# ## Full basal ganglia model in Parkinson's condition
# Finally, we'll adjust the model parameters to simulate Parkinson's disease conditions

# ![Full basal ganglia model in Parkinsonian conditions](../assets/basal_ganglia_parkinson.jpg)

# The key changes from baseline to Parkinsonian conditions are:
# 
# 1. For MSNs:
#    - Increased background excitation (`I_bg`) to 1.2519 μA·cm$^{-2}$
#    - Decreased maximal conductance for M-current (`G_M`) to 1.2 mS·cm$^{-2}$
# 
# 2. For FSIs:
#    - Decreased background excitation (`I_bg`) to 4.511 μA·cm$^{-2}$
#    - Decreased maximal conductance of FSI-MSN projection (`ḡ_inh`) by 20% to 0.48 mS·cm$^{-2}$, due to increased cholinergic tone
#    - Decreased maximal conductance of FSI-FSI projection (`weight`) to 0.2 mS·cm$^{-2}$
#    - Decreased electrical conductance (`g_elec`) to 0.075
#
# These changes reflect the loss of dopamine and increase in cholinergic tone characteristic of Parkinson's disease.

# Create bloxs with Parkinsonian parameters


@named msn = Striatum_MSN_Adam(namespace=global_ns, I_bg = 1.2519*ones(100), G_M = 1.2)
@named fsi = Striatum_FSI_Adam(namespace=global_ns, I_bg = 4.511*ones(50), weight = 0.2, g_weight = 0.075)

assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
ḡ_inh = 0.48 ## maximal conductance of FSI-MSN projection
add_edge!(g, 2, 1, Dict(:weight => ḡ_inh/7.5, :density => 0.15))
add_edge!(g, 1, 3, Dict(:weight => 2.5/33, :density => 0.33))
add_edge!(g, 3, 4, Dict(:weight => 0.3/4, :density => 0.05))
add_edge!(g, 4, 2, Dict(:weight => 0.165/4, :density => 0.1))

@named sys = system_from_graph(g)
prob = SDEProblem(sys, [], tspan, [])
ens_prob = EnsembleProblem(prob)
ens_sol = solve(ens_prob, RKMil(); dt=dt, saveat=dt, trajectories=5)

# Compute and compare power spectra for all neural populations in Parkinsonian condition against their counterparts in baseline conditions.
fig = Figure(size = (1500, 600))
ax1 = Axis(fig[1, 1], xlabel = "Frequency (Hz)", ylabel = "Power", title = "MSN (PD)", yscale = log10)
powerspectrumplot!(ax1, msn, ens_sol; state = "G", method=welch_pgram, window=hanning,
                   ylims=(1e-5, 10), alpha_label_position = (8.5, 2),
                   beta_label_position = (22, 2), gamma_label_position = (60, 2))

ax2 = Axis(fig[1, 2], xlabel = "Frequency (Hz)", ylabel = "Power", title = "FSI (PD)", yscale = log10)
powerspectrumplot!(ax2, fsi, ens_sol; state = "G", method=welch_pgram, window=hanning,
                   ylims=(1e-5, 10), alpha_label_position = (8.5, 2),
                   beta_label_position = (22, 2), gamma_label_position = (60, 2))

ax3 = Axis(fig[1, 3], xlabel = "Frequency (Hz)", ylabel = "Power", title = "GPe (PD)", yscale = log10)
powerspectrumplot!(ax3, gpe, ens_sol; state = "G", method=welch_pgram, window=hanning,
                   ylims=(1e-5, 10), alpha_start = 5, alpha_label_position = (8.5, 2),
                   beta_label_position = (22, 2), gamma_label_position = (60, 2))

ax4 = Axis(fig[1, 4], xlabel = "Frequency (Hz)", ylabel = "Power", title = "STN (PD)", yscale = log10)
powerspectrumplot!(ax4, stn, ens_sol; state = "G", method=welch_pgram, window=hanning,
                   ylims=(1e-5, 10), alpha_label_position = (8.5, 2),
                   beta_label_position = (22, 2), gamma_label_position = (60, 2))
fig

# We see the emergence of strong beta oscillations in the Parkinsonian condition (second row) respect to the baseline condition (first row) for all neural populations. This aligns with the findings of Adam et al. and reflects the pathological synchrony observed in Parkinson's disease.


# [Adam, Elie M., et al. "Deep brain stimulation in the subthalamic nucleus for Parkinson’s disease can restore dynamics of striatal networks." Proceedings of the National Academy of Sciences 119.19 (2022): e2120808119.](https://doi.org/10.1073/pnas.2120808119)