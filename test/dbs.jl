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
    transitions_inds = detect_transitions(t, stimulus; atol=0.05)
    transition_times1 = t[transitions_inds]
    transition_values1 = stimulus[transitions_inds]
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

@testset "DBS connections" begin
    # Test DBS -> single AbstractNeuronBlox
    @named dbs = DBS()
    @named n1 = HHNeuronExciBlox()
    g = MetaDiGraph()
    add_edge!(g, dbs => n1, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem

    # Test DBS -> Adam's STN
    @named stn = HHNeuronExci_STN_Adam_Blox()
    g = MetaDiGraph()
    add_edge!(g, dbs => stn, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa SDESystem

    # Test DBS -> NeuralMassBlox
    @named mass = JansenRit()
    g = MetaDiGraph()
    add_edge!(g, dbs => mass, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem

    # Test DBS -> CompositeBlox
    @named cb = CorticalBlox(namespace=:g, N_wta=2, N_exci=2, density=0.1, weight=1.0)
    g = MetaDiGraph()
    add_edge!(g, dbs => cb, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem
end