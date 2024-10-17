using Neuroblox
using Test

@testset "Detection of stimulus transitions" begin
    frequency = 0.130
    amplitude = 10.0
    pulse_width = 1
    smooth = 1e-3
    start_time = 5
    offset = -2.0
    dt = 1e-4
    tspan = (0,30)
    t = tspan[1]:dt:tspan[2]

    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    stimulus = dbs.stimulus.(t)
    transition_times1, transition_values1 = detect_transitions(t, stimulus; return_vals=true, atol=0.05)
    transition_times2 = compute_transition_times(dbs.stimulus, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    transition_values2 = compute_transition_values(transition_times2, t, stimulus)
    @test all(isapprox.(transition_times1, transition_times2, rtol=1e-3))
    @test all(isapprox.(transition_values1, transition_values2, rtol=1e-2))

    smooth = 1e-10
    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    transition_times_smoothed = compute_transition_times(dbs.stimulus, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    smooth = 0
    @named dbs = DBS(namespace=:g, frequency=frequency, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time, smooth=smooth, offset=offset)
    transition_times_non_smooth = compute_transition_times(dbs.stimulus, frequency, dt, tspan, start_time, pulse_width; atol=0.05)
    @test all(isapprox.(transition_times_smoothed, transition_times_non_smooth))
end