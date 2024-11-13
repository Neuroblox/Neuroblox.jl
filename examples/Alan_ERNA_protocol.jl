## Alan Bush and Vissani's DBS protocol tried on Elie Adam's model

using Neuroblox
using CairoMakie
using StochasticDiffEq

N_MSN = 100 ## number of Medium Spiny Neurons
N_FSI = 50 ## number of Fast Spiking Interneurons
N_GPe = 80 ## number of GPe neurons
N_STN = 40 ## number of STN neurons

global_ns = :g

@named msn = Striatum_MSN_Adam(namespace=global_ns, N_inhib = N_MSN, I_bg = 1.2519*ones(N_MSN), G_M = 1.2)
@named fsi = Striatum_FSI_Adam(namespace=global_ns, N_inhib = N_FSI, I_bg = 4.511*ones(N_FSI), weight = 0.2, g_weight = 0.075)
@named gpe = GPe_Adam(namespace=global_ns, N_inhib = N_GPe)
@named stn = STN_Adam(namespace=global_ns, N_exci = N_STN)

ḡ_FSI_MSN = 0.48 ## decreased maximal conductance of FSI-MSN projection [mS/cm^-2]
ḡ_MSN_GPe = 2.5 ## maximal conductance for MSN to GPe synapses [mS/cm^-2]
ḡ_GPe_STN = 0.3 ## maximal conductance for GPe to STN synapses [mS/cm^-2]
ḡ_STN_FSI = 0.165 ## maximal conductance for STN to FSI synapses [mS/cm^-2]

density_FSI_MSN = 0.15 ## fraction of FSIs connecting to the MSN population
density_MSN_GPe = 0.33 ## fraction of MSNs connecting to the GPe population
density_GPe_STN = 0.05 ## fraction of GPe neurons connecting to the STN population
density_STN_FSI = 0.1 ## fraction of STN neurons connecting to the FSI population

weight_FSI_MSN = ḡ_FSI_MSN / (N_FSI * density_FSI_MSN) ## normalized synaptic weight
weight_MSN_GPe = ḡ_MSN_GPe / (N_MSN * density_MSN_GPe)
weight_GPe_STN = ḡ_GPe_STN / (N_GPe * density_GPe_STN)
weight_STN_FSI = ḡ_STN_FSI / (N_STN * density_STN_FSI)

g = MetaDiGraph()
add_edge!(g, fsi => msn, weight = weight_FSI_MSN, density = density_FSI_MSN)
add_edge!(g, msn => gpe, weight = weight_MSN_GPe, density = density_MSN_GPe)
add_edge!(g, gpe => stn, weight = weight_GPe_STN, density = density_GPe_STN)
add_edge!(g, stn => fsi, weight = weight_STN_FSI, density = density_STN_FSI)


frequency = 130.0
amplitude = 600.0
pulse_width = 0.066
smooth = 1e-3
pulse_start_time = 0.008
offset = -300
pulses_per_burst = 10
bursts_per_block = 12
pre_block_time = 200.0
inter_burst_time = 200.0

@named dbs = ProtocolDBS(
                namespace=global_ns,
                frequency=frequency,
                amplitude=amplitude,
                pulse_width=pulse_width,
                smooth=smooth,
                offset=offset,
                pulses_per_burst=pulses_per_burst,
                bursts_per_block=bursts_per_block,
                pre_block_time=pre_block_time,
                inter_burst_time=inter_burst_time,
                start_time = pulse_start_time);

t_end = get_protocol_duration(dbs)
t_end = t_end + inter_burst_time

# tspan = (0.0, t_end)  # Simulation time span [ms]
tspan = (0.0, 900.0) # for testing when little RAM is available
dt = 0.001  # Time step for solving and saving [ms]

add_edge!(g, dbs => stn)

@named sys = system_from_graph(g, simplify=true)

t_ = tspan[1]:dt:tspan[2]
stimulus = dbs.stimulus.(t_)
transitions_inds = detect_transitions(t_, stimulus; atol=0.05)
transition_times = t_[transitions_inds]
transition_values = stimulus[transitions_inds]

# visualize stimulus
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "stimulus")
lines!(ax, t_, stimulus)
transition_points = scatter!(ax, transition_times, transition_values, label="transition points")
axislegend()
fig

xlims!(ax, 200, 350)
fig

xlims!(ax, 199.8, 200.3)
fig


# Creating and solving the problem
prob = SDEProblem(sys, [], tspan, [])
ens_prob = EnsembleProblem(prob)
@time ens_sol = solve(ens_prob, RKMil(); dt=dt, saveat=dt, adaptive = true, trajectories=1, abstol = 1e-3, reltol = 1e-3, tstops = transition_times);

# visualize STN average AMPA current
stn_g = meanfield_timeseries(stn, ens_sol[1], "G")
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "STN average I_AMPA ")
lines!(ax, t_, stn_g)
fig

xlims!(ax, 200, 350)
fig


stn_v = meanfield_timeseries(stn, ens_sol[1], "V")
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "STN average V ")
lines!(ax, t_, stn_v)
fig

xlims!(ax, 200, 350)
fig


fsi_v = meanfield_timeseries(fsi, ens_sol[1], "V")
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "FSI average V ")
lines!(ax, t_, fsi_v)
fig

xlims!(ax, 200, 350)
fig

msn_v = meanfield_timeseries(msn, ens_sol[1], "V")

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "MSN average V ")
lines!(ax, t_, msn_v)
fig

xlims!(ax, 200, 350)
fig

gpe_v = meanfield_timeseries(gpe, ens_sol[1], "V")

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "GPe average V ")
lines!(ax, t_, gpe_v)
fig

xlims!(ax, 200, 350)
fig