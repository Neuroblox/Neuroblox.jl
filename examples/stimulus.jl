using Neuroblox
using OrdinaryDiffEq
using CairoMakie
using Random
Random.seed!(1)


# Square pulse stimulus
global_ns = :g
@named stim = DBS(
                namespace=global_ns,
                frequency=100.0, # in Hz
                amplitude=200.0, # in arbitrary units, depends on how the stimulus is used in the model
                pulse_width=0.5, # in ms
                offset=0.0,
                start_time=5.0, # in ms
                smooth=0.0
                );


tspan = (0.0, 100.0)
dt = 0.001

time = tspan[1]:dt:tspan[2]
stimulus = stim.stimulus.(time)


fig = Figure();
ax1 = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "stimulus")
lines!(ax1, time, stimulus)
fig


# Smoothed square pulse stimulus
@named stim_smooth = DBS(
                namespace=global_ns,
                frequency=100.0,
                amplitude=200.0,
                pulse_width=0.5,
                offset=0.0,
                start_time=5.0,
                smooth=1e-3
                );

smooth_stimulus = stim_smooth.stimulus.(time)


fig = Figure();
ax1 = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "stimulus")
lines!(ax1, time, stimulus)
xlims!(ax1, 4.9, 5.6)

# fig = Figure();
ax2 = Axis(fig[2,1]; xlabel = "time (ms)", ylabel = "stimulus")
lines!(ax2, time, smooth_stimulus)
xlims!(ax2, 4.9, 5.6)

fig

# Compute transition points to aid differential equation solving
transition_inds_1 = detect_transitions(time, stimulus; atol=0.05)
transition_times_1 = time[transition_inds_1]
transition_values_1 = stimulus[transition_inds_1]

transition_inds_2 = detect_transitions(time, smooth_stimulus; atol=0.05)
transition_times_2 = time[transition_inds_2]
transition_values_2 = smooth_stimulus[transition_inds_2]

scatter!(ax1, transition_times_1, transition_values_1)
scatter!(ax2, transition_times_2, transition_values_2)
fig



# It is also possible to create a stimulus protocol as follows:
frequency = 20.0
amplitude = 1.0
pulse_width = 20.0
smooth = 3e-4
pulse_start_time = 0.01
offset = 0
pulses_per_burst = 3
bursts_per_block = 2
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
tspan = (0.0, t_end)
dt = 0.001

time = tspan[1]:dt:tspan[2]
stimulus = dbs.stimulus.(time)
transitions_inds = detect_transitions(time, stimulus; atol=0.001)

transition_times = time[transitions_inds]
transition_values = stimulus[transitions_inds]


fig = Figure();
ax1 = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "stimulus")
lines!(ax1, time, stimulus)
fig

scatter!(ax1, transition_times, transition_values)
xlims!(ax1, 199.9, 200.2)
fig



# Single HH neuron
@named nn = HHNeuronExciBlox(I_bg=0.4)

# Connect the stimulus to the neuron
g = MetaDiGraph()
add_edge!(g, dbs => nn, weight = 10.0)

# Solve the system equations
@named sys = system_from_graph(g)
prob = ODEProblem(sys, [], tspan, [])
@time sol = solve(prob, Vern7(), saveat=dt, tstops = transition_times); # Note: Providing transition times is not necessary for this example but may be required for much shorter pulses

# Plot the membrane potential
v = voltage_timeseries(nn, sol)
fig = Figure();
ax1 = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mV)")
lines!(ax1, sol.t, v)

ax2 = Axis(fig[2,1]; xlabel = "time (ms)", ylabel = "Stimulus (μA/cm²)")
lines!(ax2, sol.t, stimulus)
fig