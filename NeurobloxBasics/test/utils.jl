using NeurobloxBasics
using OrdinaryDiffEqTsit5
using StochasticDiffEq
using Statistics
using SparseArrays
using Peaks: argmaxima
using Test

@testset "Voltage timeseries [LIFExciNeuron]" begin
    global_ns = :g 
    @named n = LIFExciNeuron(; namespace = global_ns)

    g = GraphSystem()
    add_node!(g, n)

    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())

    @test all(sol[n.V] .== NeurobloxBase.voltage_timeseries(n, sol))
end

@testset "Voltage timeseries [Vector{<:AbstractNeuron}]" begin
    global_ns = :g # global namespace
    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(3, (0, 200))
            n1 = LIFExciNeuron()
            n2 = LIFExciNeuron()
            n3 = LIFInhNeuron()
        end
        @connections begin
            n1 => n2, [weight=1]
            n1 => n3, [weight=1]
            s  => n1, [weight=1]
        end
    end

    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())

    V = hcat(sol[n1.V], sol[n2.V], sol[n3.V])

    @test all(V .== NeurobloxBase.voltage_timeseries([n1, n2, n3], sol))
end

@testset "Voltage timeseries + Composite average [LIFExciCircuitz]" begin
    global_ns = :g 
    tspan = (0, 200)
    V_reset = -55

    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(3, tspan)
            n = LIFExciCircuit(; V_reset,  N_neurons = 3, weight=1)
        end
        @connections begin
            s => n, [weight=1]
        end
    end
    prob = ODEProblem(g, [], (0, 200.0))
    sol = solve(prob, Tsit5())
    
    V = hcat(sol[n.neurons[1].V], sol[n.neurons[2].V], sol[n.neurons[3].V])
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
    @graph g begin
        @nodes begin
            s = PoissonSpikeTrain(3, (0, 200))
            n = LIFExciNeuron(;)
            cb = LIFExciCircuit(; N_neurons = 3, weight=1)
        end
        @connections begin
            s => n, [weight=1]
            s => cb, [weight=1]
        end
    end
    prob = ODEProblem(g, [], (0, 200.0))
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
