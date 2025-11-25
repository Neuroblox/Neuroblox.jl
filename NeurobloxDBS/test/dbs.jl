using NeurobloxDBS
using Test

@testset "Detection of stimulus transitions" begin
    frequency = 130.0
    amplitude = 10.0
    pulse_width = 1
    smooth = 1e-3
    start_time = 5
    offset = -2.0
    dt = 1e-4
    tspan = (0,30)
    t = tspan[1]:dt:tspan[2]

    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    stim_fun = get_stimulus_function(dbs)
    stimulus = stim_fun.(t)
    
    transitions_inds = detect_transitions(t, stimulus; atol=0.05)
    transition_times1 = t[transitions_inds]
    transition_values1 = stimulus[transitions_inds]
    transition_times2 = compute_transition_times(stim_fun, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    transition_values2 = compute_transition_values(transition_times2, t, stimulus)
    @test all(isapprox.(transition_times1, transition_times2, rtol=1e-3))
    @test all(isapprox.(transition_values1, transition_values2, rtol=1e-2))

    smooth = 1e-10
    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    stim_fun = get_stimulus_function(dbs)
    transition_times_smoothed = compute_transition_times(stim_fun, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    smooth = 0
    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    stim_fun = get_stimulus_function(dbs)
    transition_times_non_smooth = compute_transition_times(stim_fun, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    @test all(isapprox.(transition_times_smoothed, transition_times_non_smooth))
end

@testset "DBS connections" begin
    # Test DBS -> Adam's STN
    @named dbs = DBS()
    @named stn = HHNeuronExci_STN_Adam()
    g = MetaDiGraph()
    add_edge!(g, dbs => stn, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa SDESystem
end

@testset "ProtocolDBS" begin
    frequency=100.0  # Hz
    amplitude=2.5
    pulse_width=0.066
    offset=0.0
    pulses_per_burst=5
    bursts_per_block=3
    pre_block_time=100.0
    inter_burst_time=50.0
    smooth=1e-3
    start_time = 0.008

    dbs = ProtocolDBS(;
        name=:test,
        frequency=frequency,
        amplitude=amplitude,
        pulse_width=pulse_width,
        offset=offset,
        pulses_per_burst=pulses_per_burst,
        bursts_per_block=bursts_per_block,
        pre_block_time=pre_block_time,
        inter_burst_time=inter_burst_time,
        smooth=smooth,
        start_time=start_time
    )

    # Test pre-block period
    ts = 0.0:0.1:pre_block_time-1
    stim_fun = get_stimulus_function(dbs)
    @test all(stim_fun.(ts) .== offset)

    # Should see pulses within burst
    period = 1000.0/frequency  # period in ms from 100Hz
    ts = [pre_block_time + pulse_width/2 + i*period for i in 0:pulses_per_burst-1]
    @test all(stim_fun.(ts) .≈ amplitude)

    # Test inter-burst period
    t_between = pre_block_time + pulses_per_burst*period + inter_burst_time/2  # middle of inter-burst
    @test stim_fun(t_between) == offset

    # Test protocol duration
    duration = get_protocol_duration(dbs)
    expected =  pre_block_time +
                bursts_per_block * (pulses_per_burst * period + inter_burst_time) -
                inter_burst_time  # subtract last inter_burst
    @test duration == expected

    # Test number of stimulus
    t_end = duration + inter_burst_time
    tspan = (0.0, t_end)
    ts = tspan[1]:0.001:tspan[2]
    stim_fun = get_stimulus_function(dbs)
    stimulus = stim_fun.(ts)
    transitions_inds = detect_transitions(ts, stimulus; atol=0.005)
    transition_times = ts[transitions_inds]
    transition_values = stimulus[transitions_inds]

    @test sum(isapprox.(transition_values, amplitude, atol=0.1)) == 2*bursts_per_block*pulses_per_burst
    @test sum(isapprox.(transition_values, offset, atol=0.1)) == 2*bursts_per_block*pulses_per_burst
end

