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

    @test all(sol[ss.n.V] .== Neuroblox.voltage_timeseries(sol, n))
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

    @test all(isequal(V, Neuroblox.voltage_timeseries(sol, n)))

    V_filtered = map(eachrow(V)) do V_t
        v = filter(!isnan, V_t)
        mean(v)
    end
    
    @test all(isequal(V_filtered, Neuroblox.average_voltage_timeseries(sol, n)))
end
