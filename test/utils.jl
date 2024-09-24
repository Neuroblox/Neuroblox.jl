using Neuroblox
using DifferentialEquations
using Statistics

@testset "Voltage timeseries [LIFExciNeuron]" begin
    global_ns = :g 
    @named n = LIFExciNeuron(; namespace = global_ns)

    g = MetaDiGraph()
    add_blox!(g, n)

    sys = system_from_graph(g; name = global_ns)
    ss = structural_simplify(sys)
    prob = ODEProblem(ss, [], (0, 200.0))
    sol = solve(prob, Tsit5())

    @test all(sol[ss.n.V] .== Neuroblox.voltage_timeseries(n, sol))
end

@testset "Voltage timeseries + Composite average [LIFExciCircuitBloxz]" begin
    global_ns = :g 
    tspan = (0, 200)
    V_reset = -55
    
    @named s = PoissonSpikeTrain(3, tspan; namespace = global_ns)
    @named n = LIFExciCircuitBlox(; V_reset, namespace = global_ns, N_neurons = 3, weight=1)
    
    neurons = [s, n]
    
    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)
    
    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name = global_ns)
    ss = structural_simplify(sys)
    prob = ODEProblem(ss, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    
    V = hcat(sol[ss.n.neuron1.V], sol[ss.n.neuron2.V], sol[ss.n.neuron3.V])
    V[V .== V_reset] .= NaN
    
    V_nb = Neuroblox.voltage_timeseries(n, sol)
    Neuroblox.replace_refractory!(V_nb, n, sol)
    @test all(isequal(V, V_nb))
    
    V_filtered = map(eachrow(V)) do V_t
        v = filter(!isnan, V_t)
        mean(v)
    end
    
    @test all(isequal(V_filtered, Neuroblox.meanfield_timeseries(n, sol)))
end

@testset "Powerspectrum" begin

    # AbstractNeuronBlox
    nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=3, freq=80)
    nn2 = HHNeuronExciBlox(name=Symbol("nrn2"), I_bg=2, freq=1)
    assembly = [nn1, nn2]
    
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g, 1, 2, :weight, 1)
    
    @named neuron_net = system_from_graph(g)
    prob = ODEProblem(structural_simplify(neuron_net), [], (0.0, 200), [])
    sol = solve(prob, Vern7(), saveat=0.01)
    ps = powerspectrum(nn1, sol, "V")
    
    peak_freq = ps.freq[argmax(ps.power[3:end])+2]
    @test isapprox(peak_freq, 80, atol=1)
    
    # test Welch periodogram and windows
    ps2 = powerspectrum(nn1, sol, "V", window = hamming) 
    ps3 = powerspectrum(nn1, sol, "V", method = welch_pgram, window = hanning) 
    peak_freq2 = ps2.freq[argmax(ps2.power[3:end])+2]
    peak_freq3 = ps3.freq[argmax(ps3.power[3:end])+2]
    @test isapprox(peak_freq, peak_freq2, atol=0.001)
    @test isapprox(peak_freq, peak_freq3, atol=1)

    # test resampling
    sol = solve(prob, Vern7())
    ps4 = powerspectrum(nn1, sol, "V"; sampling_rate=0.01)
    peak_freq4 = ps4.freq[argmax(ps4.power[3:end])+2]
    @test peak_freq4 ≈ peak_freq

    # CompositeBlox
    global_ns = :g 
    @named LC = NextGenerationEIBlox(;namespace=global_ns)
    @named cb = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
    assembly = [LC, cb]
    g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)
    add_edge!(g,1,2, :weight, 44)
    neuron_net = system_from_graph(g; name=global_ns)
    prob = ODEProblem(structural_simplify(neuron_net), [], (0.0, 1.0), [])
    sol = solve(prob, Vern7())
    ps = powerspectrum(cb, sol, "V")
    ps2 = powerspectrum(cb, sol)
    @test all(ps.power .== ps2.power)
end