using Neuroblox
using OrdinaryDiffEqVerner
using StochasticDiffEq
using Statistics
using SparseArrays
using Peaks: argmaxima

@testset "Powerspectrum" begin

    # AbstractNeuron
    @named nn = HHNeuronExci(I_bg=0.4)
    @named dbs = DBS(
                frequency=80.0,
                amplitude=10.0,
                pulse_width=0.5,
                offset=0.0,
                start_time=0.0,
                smooth=0.0);

    g = MetaDiGraph()
    add_edge!(g, dbs => nn, weight = 10.0)

    @named sys = system_from_graph(g)
    prob = ODEProblem(sys, [], (0.0, 500), [])
    sol = solve(prob, Vern7(), saveat=0.01)
    ps = powerspectrum(nn, sol, "V")

    n = findfirst(ps.freq .> 100)
    ind = only(argmaxima(ps.power[1:n], 4))
    peak_freq = ps.freq[ind]

    @test isapprox(peak_freq, 80, atol=0.5)
    
    # test Welch periodogram and windows
    ps2 = powerspectrum(nn, sol, "V", window = hamming) 
    ps3 = powerspectrum(nn, sol, "V", method = welch_pgram, window = hanning) 

    n = findfirst(ps2.freq .> 100)
    ind = only(argmaxima(ps2.power[1:n], 4))
    peak_freq2 = ps2.freq[ind]

    n = findfirst(ps3.freq .> 100)
    ind = only(argmaxima(ps3.power[1:n], 2))
    peak_freq3 = ps3.freq[ind]

    @test isapprox(peak_freq, peak_freq2, atol=0.001)
    @test isapprox(peak_freq, peak_freq3, atol=1)

    # test resampling
    sol = solve(prob, Vern7())
    ps4 = powerspectrum(nn, sol, "V"; sampling_rate=0.01)

    n = findfirst(ps4.freq .> 100)
    ind = only(argmaxima(ps4.power[1:n], 4))
    peak_freq4 = ps4.freq[ind]

    @test isapprox(peak_freq, peak_freq4, atol=0.5)

    # AbstractComposite
    global_ns = :g 
    @named LC = NextGenerationEI(;namespace=global_ns)
    @named cb = Cortical(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    assembly = [LC, cb]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(neuron_net, [], (0.0, 1.0), [])
    sol = solve(prob, Vern7(), saveat=0.1)
    ps = powerspectrum(cb, sol, "V")
    ps2 = powerspectrum(cb, sol)
    @test all(ps.power .== ps2.power)
end

@testset "DBS stimulus to neuron connections" begin
    @named dbs = DBS(namespace=:g)
    @named n1 = HHNeuronExci()
    g = MetaDiGraph()
    add_edge!(g, dbs => n1, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    prob = ODEProblem(sys, [], (0,1), [])
    stim_fun = get_stimulus_function(dbs)
    stim_fun2 = get_stimulus_function(prob)
    stim_fun3 = get_stimulus_function(sys)
    @test stim_fun == stim_fun2
    @test stim_fun == stim_fun3

    stim2 = SquareStimulus(200, 2.5, 0.0, 0.0, 0.066, 1e-4)
    prob2 = remake(prob; p=[sys.dbs.stimulus => stim2])
    stim_fun4 = get_stimulus_function(prob2)
    @test stim_fun4.frequency_khz == 0.2

    @named dbs = ProtocolDBS(namespace=:g)
    @named n1 = HHNeuronExci()
    g = MetaDiGraph()
    add_edge!(g, dbs => n1, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    prob = ODEProblem(sys, [], (0,1), [])
    stim_fun = get_stimulus_function(dbs)
    stim_fun2 = get_stimulus_function(prob)
    stim_fun3 = get_stimulus_function(sys)
    @test stim_fun == stim_fun2
    @test stim_fun == stim_fun3

    t1 = get_protocol_duration(dbs)
    t2 = get_protocol_duration(prob)
    t3 = get_protocol_duration(sys)
    @test t1 == t2
end

@testset "DBS + Blox system creation" begin
    # Test DBS -> single AbstractNeuron
    @named dbs = DBS()
    @named n1 = HHNeuronExci()
    g = MetaDiGraph()
    add_edge!(g, dbs => n1, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem

    # Test DBS -> AbstractNeuralMass
    @named mass = JansenRit()
    g = MetaDiGraph()
    add_edge!(g, dbs => mass, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem

    # Test DBS -> AbstractComposite
    @named cb = Cortical(namespace=:g, N_wta=2, N_exci=2, density=0.1, weight=1.0)
    g = MetaDiGraph()
    add_edge!(g, dbs => cb, weight = 1.0)
    sys = system_from_graph(g; name=:test)
    @test sys isa ODESystem
end
