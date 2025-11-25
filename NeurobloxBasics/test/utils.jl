using NeurobloxBasics
using OrdinaryDiffEqTsit5
using StochasticDiffEq
using Statistics
using SparseArrays
using Peaks: argmaxima

@testset "Voltage timeseries [LIFExciNeuron]" begin
    global_ns = :g 
    @named n = LIFExciNeuron(; namespace = global_ns)

    g = MetaDiGraph()
    add_blox!(g, n)

    sys = system_from_graph(g; name = global_ns)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())

    @test all(sol[sys.n.V] .== NeurobloxBase.voltage_timeseries(n, sol))
end

@testset "Voltage timeseries [Vector{<:AbstractNeuron}]" begin
    global_ns = :g # global namespace
    @named s = PoissonSpikeTrain(3, (0, 200); namespace = global_ns)
    @named n1 = LIFExciNeuron(; namespace = global_ns)
    @named n2 = LIFExciNeuron(; namespace = global_ns)
    @named n3 = LIFInhNeuron(; namespace = global_ns)

    neurons = [n1, n2, n3]
    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)

    add_edge!(g, n1 => n2; weight=1)
    add_edge!(g, n1 => n3; weight=1)
    add_edge!(g, s => n1; weight=1)

    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())

    V = hcat(sol[sys.n1.V], sol[sys.n2.V], sol[sys.n3.V])

    @test all(V .== NeurobloxBase.voltage_timeseries([n1, n2, n3], sol))
end

@testset "Voltage timeseries + Composite average [LIFExciCircuitz]" begin
    global_ns = :g 
    tspan = (0, 200)
    V_reset = -55
    
    @named s = PoissonSpikeTrain(3, tspan; namespace = global_ns)
    @named n = LIFExciCircuit(; V_reset, namespace = global_ns, N_neurons = 3, weight=1)
    
    neurons = [s, n]
    
    g = MetaDiGraph()
    add_blox!.(Ref(g), neurons)
    
    add_edge!(g, 1, 2, Dict(:weight => 1))
    
    sys = system_from_graph(g; name = global_ns)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    
    V = hcat(sol[sys.n.neuron1.V], sol[sys.n.neuron2.V], sol[sys.n.neuron3.V])
    V[V .== V_reset] .= NaN
    
    V_nb = NeurobloxBase.voltage_timeseries(n, sol)
    NeurobloxBase.replace_refractory!(V_nb, n, sol)
    @test all(isequal(V, V_nb))
    
    V_filtered = map(eachrow(V)) do V_t
        v = filter(!isnan, V_t)
        mean(v)
    end
    
    @test all(isequal(V_filtered, NeurobloxBase.meanfield_timeseries(n, sol)))
end

@testset "Spike detection [LIFExciNeuron and LIFExciCircuit]" begin
    global_ns = :g # global namespace
    @named s = PoissonSpikeTrain(3, (0, 200); namespace = global_ns)
    @named n = LIFExciNeuron(; namespace = global_ns)
    @named cb = LIFExciCircuit(; namespace = global_ns, N_neurons = 3, weight=1)
    
    g = MetaDiGraph()
    
    add_edge!(g, s => n; weight=1)
    add_edge!(g, s => cb; weight=1)
    
    sys = system_from_graph(g; name=global_ns)
    prob = ODEProblem(sys, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    
    spikes_n = detect_spikes(n, sol)
    spikes_cb = detect_spikes(cb, sol)
    
    @test !iszero(nnz(spikes_n))
    @test !iszero(nnz(spikes_cb))
    
    spikes_n = detect_spikes(n, sol; threshold = 10)
    spikes_cb = detect_spikes(cb, sol; threshold = 10)
    
    @test iszero(nnz(spikes_n))
    @test iszero(nnz(spikes_cb)) 
end
