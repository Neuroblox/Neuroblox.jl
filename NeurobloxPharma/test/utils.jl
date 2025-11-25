using NeurobloxPharma
using OrdinaryDiffEqDefault
using Statistics

@testset "Inter-Spike Intervals [AbstractNeuron and Vector{<:AbstractNeuron}]" begin

    @named hh1 = HHNeuronExci(; I_bg=0.4)
    tspan = (0.0, 1000.0)
    sys = system(hh1)
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob; saveat=0.05)

    ISIs = inter_spike_intervals(hh1, sol; threshold=0)
    freq = 1e3/mean(ISIs)

    @test isapprox(freq, 12, atol=0.5) 

    neurons = [hh1, hh1]
    ISIs = inter_spike_intervals(neurons, sol; threshold=0)

    for i in eachindex(neurons)
        freq = 1e3/mean(ISIs[:,i])
        @test isapprox(freq, 12, atol=0.5) 
    end
end
