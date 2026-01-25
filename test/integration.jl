using Neuroblox
using OrdinaryDiffEqVerner
using StochasticDiffEq
using Statistics
using SparseArrays
using Peaks: argmaxima
using ReferenceTests
using CairoMakie

@testset "Powerspectrum" begin
    g = @graph begin
        @nodes begin
            nn = HHNeuronExci(I_bg = 0.4)
            dbs = DBS(frequency = 80.0,
                      amplitude = 10.0,
                      pulse_width = 0.5,
                      offset = 0.0,
                      start_time = 0.0,
                      smooth = 0.0);
        end
        @connections begin
            dbs => nn, [weight = 10.0]
        end
    end
    prob = ODEProblem(g, [], (0.0, 500), [])
    sol = solve(prob, Vern7(), saveat=0.01)
    ps = powerspectrum(nn, sol, "V")
    plt = powerspectrumplot(nn, sol)
    @test_reference "plots/powerspectrum.png" plt by=psnr_equality(20)

    n = findfirst(ps.freq .> 100)
    ind = only(argmaxima(ps.power[1:n], 4))
    peak_freq = ps.freq[ind]

    @test isapprox(peak_freq, 80, atol=0.5)
    
    # test Welch periodogram and windows
    ps2 = powerspectrum(nn, sol, "V", window = hamming) 
    plt2 = powerspectrumplot(nn, sol; window = hamming)
    @test_reference "plots/powerspectrum2.png" plt2 by=psnr_equality(30)
    ps3 = powerspectrum(nn, sol, "V", method = welch_pgram, window = hanning)
    plt3 = powerspectrumplot(nn, sol; method = welch_pgram, window = hamming)
    @test_reference "plots/powerspectrum3.png" plt3 by=psnr_equality(30)

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
    plt4 = powerspectrumplot(nn, sol; sampling_rate = 0.01)
    @test_reference "plots/powerspectrum4.png" plt4 by=psnr_equality(30)

    n = findfirst(ps4.freq .> 100)
    ind = only(argmaxima(ps4.power[1:n], 4))
    peak_freq4 = ps4.freq[ind]

    @test isapprox(peak_freq, peak_freq4, atol=0.5)

    # AbstractComposite
    g = @graph begin
        @nodes begin
            LC = NextGenerationEI(; namespace = :g)
            cb = Cortical(N_wta=2, N_exci=2, namespace= :g, density=0.1, weight=1)
        end
        @connections begin
            LC => cb, [weight = 44]
        end
    end
    prob = ODEProblem(g, [], (0.0, 1.0), [])
    sol = solve(prob, Vern7(), saveat=0.1)

    ps = powerspectrum(cb, sol, "V")
    ps2 = powerspectrum(cb, sol)
    @test all(ps.power .== ps2.power)
end

@testset "DBS stimulus to neuron connections" begin
    g = @graph begin
        @nodes begin
            dbs = DBS(namespace = :g)
            n1 = HHNeuronExci()
        end
        @connections begin
            dbs => n1, [weight = 1.0]
        end
    end

    prob = ODEProblem(g, [], (0,1), [])
    stim_fun = get_stimulus_function(dbs)
    stim_fun2 = get_stimulus_function(prob)
    stim_fun3 = get_stimulus_function(g)
    @test stim_fun isa SquareStimulus
    @test stim_fun == stim_fun2
    @test stim_fun == stim_fun3

    prob2 = remake(prob; p = [:stimulus_DBSConnection_g₊dbs_n1 => SquareStimulus(200, 2.5, 0.0, 0.0, 0.066, 1e-4)])

    stim_fun4 = get_stimulus_function(prob2)
    @test stim_fun4.frequency_khz == 0.2

    g = @graph begin
        @nodes begin
            dbs = ProtocolDBS(namespace = :g)
            n1 = HHNeuronExci()
        end
        @connections begin
            dbs => n1, [weight = 1.0]
        end
    end

    prob = ODEProblem(g, [], (0,1), [])
    stim_fun = get_stimulus_function(dbs)
    stim_fun2 = get_stimulus_function(prob)
    stim_fun3 = get_stimulus_function(g)
    @test stim_fun == stim_fun2
    @test stim_fun == stim_fun3

    t1 = get_protocol_duration(dbs)
    t2 = get_protocol_duration(prob)
    t3 = get_protocol_duration(g)
    @test t1 == t2
end

@testset "DBS + Blox system creation" begin
    g = @graph begin
        @nodes begin
            dbs = ProtocolDBS(namespace = :g)
            n1 = HHNeuronExci()
            mass = JansenRit()
            cb = Cortical(namespace=:g, N_wta=2, N_exci=2, density=0.1, weight=1.0)
        end
        @connections begin
            dbs => n1, [weight = 1.0]
            dbs => mass, [weight = 1.0]
            dbs => cb, [weight = 1.0]
        end
    end

    @test g.flat_graph isa PartitioningGraphSystem
end
