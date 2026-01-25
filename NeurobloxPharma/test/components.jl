using NeurobloxPharma
using OrdinaryDiffEqVerner, OrdinaryDiffEqTsit5
using ReferenceTests, CairoMakie
using StableRNGs

rng = StableRNG(2026)

@testset "HH Neuron excitatory & inhibitory network" begin
    # Adjacency matrix : 
    #adj = [0 1 0
    #       0 0 1
    #       0.2 0 0]
    @graph g begin
        @nodes begin
            nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=3)
            nn2 = HHNeuronExci(name=Symbol("nrn2"), I_bg=2)
            nn3 = HHNeuronInhib(name=Symbol("nrn3"), I_bg=2)
        end
        @connections begin
            nn1 => nn2, [weight=1]
            nn2 => nn3, [weight=5.5]
            nn3 => nn1, [weight=0.2]
        end
    end
    prob = ODEProblem(g, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success

    plt = stackplot([nn1, nn2, nn3], sol)
    @test_reference "plots/hh_network.png" plt by=psnr_equality(40)
end

@testset "NGNMM_theta connected to neuron" begin
    @graph g begin
        @nodes begin
            LC = NextGenerationEI(; Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
            nn = HHNeuronExci(;)
        end
        @connections begin
            LC => nn, [weight=44]
        end
    end
    prob = ODEProblem(g, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success

    plt = stackplot([nn], sol)
    @test_reference "plots/ngnmm_neuron.png" plt by=psnr_equality(40)
end

@testset "NGNMM_theta connected to Cortical" begin
    @graph g begin
        @nodes begin
            LC = NGNMM_theta(;Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
            cb = Cortical(N_wta=2, N_exci=2, density=0.1, weight=1, rng=StableRNG(12345))
        end
        @connections begin
            LC => cb, [weight=44]
        end
    end 
    prob = ODEProblem(g, [], (0.0, 2), [])
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success

    plt = meanfield(cb, sol)
    @test_reference "plots/ngnmm_cortical.png" plt by=psnr_equality(40)
end

@testset "WinnerTakeAll" begin
    N_exci = 5
    rng = StableRNG(6789)
    @named wta= WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci)
    g = wta.graph
    prob = ODEProblem(g,[],(0,10))
    sol = solve(prob, Vern7(), saveat=0.1)

    @test sol.retcode == ReturnCode.Success 

    plt = stackplot(wta, sol)
    @test_reference "plots/wta.png" plt by=psnr_equality(40)
end

@testset "WinnerTakeAll network" begin
    N_exci = 5
    rng = StableRNG(5678)
    @graph g begin
        @nodes begin
            wta1 = WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci)
            wta2 = WinnerTakeAll(;I_bg=5.0*rand(rng, N_exci), N_exci)
        end
        @connections begin
            wta1 => wta2, [weight=1, density=0.5, rng=rng]
        end
    end
    prob = ODEProblem(g, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success 

    plt = stackplot([wta1, wta2], sol)
    @test_reference "plots/wta_network.png" plt by=psnr_equality(40)
end

@testset "Cortical" begin
    @named cb = Cortical(N_wta=6, N_exci=5, density=0.1, weight=1, rng=StableRNG(123458))
    g = cb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = meanfield(cb, sol)
    @test_reference "plots/cortical.png" plt by=psnr_equality(40)
end

@testset "Striatum" begin
    @named str_scb = Striatum(N_inhib=2)
    g = str_scb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = meanfield(str_scb, sol)
    @test_reference "plots/striatum.png" plt by=psnr_equality(40)
end

@testset "GPi" begin
    @named gpi_scb = GPi(N_inhib=2)
    g = gpi_scb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = stackplot(gpi_scb, sol)
    @test_reference "plots/gpi.png" plt by=psnr_equality(40)
end

@testset "GPe" begin
    @named gpe_scb = GPe(N_inhib=2)
    g = gpe_scb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = stackplot(gpe_scb, sol)
    @test_reference "plots/gpe.png" plt by=psnr_equality(40)
end

@testset "STN" begin
    @named stn_scb = STN(N_exci=2)
    g = stn_scb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = stackplot(stn_scb, sol)
    @test_reference "plots/stn.png" plt by=psnr_equality(40)
end

@testset "Thalamus" begin
    @named thal_scb = Thalamus(N_exci=2)
    g = thal_scb.graph
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7(), saveat=0.5)
    @test sol.retcode == ReturnCode.Success

    plt = stackplot(thal_scb, sol)
    @test_reference "plots/thalamus.png" plt by=psnr_equality(40)
end

@testset "Cortical-ImageStimulus network" begin
    fn = joinpath(@__DIR__, "../../examples/image_example.csv")
    @graph g begin
        @nodes begin
            stim = ImageStimulus(fn; t_stimulus=1, t_pause=0.5)
            cb = Cortical(N_wta=2, N_exci=2, density=0.1, weight=1, rng=StableRNG(123459))
        end
        @connections begin
            stim => cb, [weight=1]
        end
    end
    prob = ODEProblem(g, [], (0, 2))
    sol = solve(prob, Vern7())
    @test sol.retcode == ReturnCode.Success

    plt = stackplot(cb, sol)
    @test_reference "plots/cortical_image.png" plt by=psnr_equality(40)
end

@testset "Cortical-Cortical network" begin
    @graph g begin
        @nodes begin
            cb1 = Cortical(N_wta=2, N_exci=2, density=0.1, weight=1, rng=StableRNG(123460))
            cb2 = Cortical(N_wta=3, N_exci=3, density=0.1, weight=1, rng=StableRNG(123461))
        end
        @connections begin
            cb1 => cb2, [weight=1, density=0.1]
        end
    end
    prob = ODEProblem(g, [], (0,2))
    sol = solve(prob, Vern7(), saveat=0.1)
    @test sol.retcode == ReturnCode.Success

    plt = stackplot([cb1, cb2], sol)
    @test_reference "plots/cort_cort.png" plt by=psnr_equality(40)
end

@testset "Cortical & subcortical components network" begin
    @graph g begin
        @nodes begin
            cb1 = Cortical(N_wta=3, N_exci=3, density=0.1, weight=1, rng=StableRNG(123462))
            cb2 = Cortical(N_wta=2, N_exci=2, density=0.1, weight=1, rng=StableRNG(123463))
            str1 = Striatum(N_inhib=2)
            gpi1 = GPi(N_inhib=2)
            thal1 = Thalamus(N_exci=2)
        end
        @connections begin
            cb1 => cb2, [weight=1, density=0.1]
            cb2 => str1, [weight=1, density=0.1]
            str1 => gpi1, [weight=1, density=0.1]
            gpi1 => thal1, [weight=1, density=0.1]
            thal1 => cb2, [weight=1, density=0.1]
        end
    end
    prob = ODEProblem(g, [], (0,2))
    sol = solve(prob, Tsit5())

    fig, ax, plt = meanfield(cb1, sol)
    meanfield!(ax, cb2, sol)
    meanfield!(ax, str1, sol)
    meanfield!(ax, gpi1, sol)
    meanfield!(ax, thal1, sol)
    @test_reference "plots/cort_subcort.png" fig by=psnr_equality(40)
end

@testset "Morandi: " for T_NMDAR ∈ (MoradiNMDAR, MoradiFullNMDAR) 
    @testset "Clamp" begin
        # Based on Moradi et al 2013
        @graph g begin
            @nodes begin
                src = VoltageClampSource([(t=50, V=-80), (t=100, V=40)])
                nmda = T_NMDAR(; τ_g=50)
            end
            @connections src => nmda, [weight = 1.0]
        end
        prob = ODEProblem(g, [nmda.A => 4.0, nmda.B => 4.0, src.V => 40.0], (0.0, 500.0))
        sol = solve(prob, Tsit5(), saveat=0.05)
        fig = Figure()
        ax = Axis(fig[1,1]; ylabel="NMDA Current", xlabel="Time (ms)")
        lines!(ax, sol.t, sol[nmda.I])
        @test_reference "plots/$(T_NMDAR)_I_clamp.png" fig by=psnr_equality(25)
    end
    @testset "Neurons" begin
        # This isn't based on anything
        @graph g begin
            @nodes begin
                n1 = HHNeuronExci(I_bg=1.0)
                n2 = HHNeuronExci(I_bg=1.50)
                nmda = T_NMDAR()
            end
            @connections begin
                n1 => n2, [weight = 1.0, synapse=nmda]
            end
        end
        prob = ODEProblem(g, [], (0.0, 100.0))
        sol = solve(prob, Tsit5(), saveat=0.05)
        fig = Figure()
        ax = Axis(fig[1,1]; ylabel="NMDA Current", xlabel="Time (ms)")
        lines!(ax, sol.t, sol[nmda.I])
        @test_reference "plots/$(T_NMDAR)_neuron_conn.png" fig by=psnr_equality(25)
    end
end

@testset "NMDA/AMPA synapses" begin
    @graph g begin
        @nodes begin
            syn = Glu_AMPA_Synapse(; name=:ampa)
            ne = HHNeuronExci(; I_bg=5)
            ni = HHNeuronInhib()
        end
        @connections begin
            ne => ni , [weight=1, synapse=syn]
            ni => ne , [weight=1]
        end
    end

    prob = ODEProblem(g, [], (0, 200))
    sol = solve(prob, Vern7())
    fre_ampa = firing_rate(ne, sol; threshold=0)
    fri_ampa = firing_rate(ni, sol; threshold=0)

    @graph g begin
        @nodes begin
            ne = HHNeuronExci(; I_bg=5)
            ni = HHNeuronInhib()
        end
        @connections begin
            ne => ni , [weight=1, synapse_type=NMDA_Synapse]
            ni => ne , [weight=1]
        end
    end

    prob = ODEProblem(g, [], (0, 500))
    sol = solve(prob, Vern7())
    fre_nmda = firing_rate(ne, sol; threshold=0)
    fri_nmda = firing_rate(ni, sol; threshold=0)

    @test fre_nmda > fre_ampa
    @test fri_nmda < fri_ampa
end

@testset "GABA A/B synapses" begin
    @graph g begin
        @nodes begin
            ne = HHNeuronExci(; I_bg=8)
            ni = HHNeuronInhib(; I_bg=1)
            synapse = GABA_A_Synapse()
        end
        @connections begin
            ni => ne , [weight=1, synapse_type=GABA_A_Synapse]
        end
    end

    prob = ODEProblem(g, [], (0, 1000))
    sol = solve(prob, Vern7())
    fre_gabaa = firing_rate(ne, sol; threshold=0)
    fri_gabaa = firing_rate(ni, sol; threshold=0)

    @graph g begin
        @nodes begin
            ne = HHNeuronExci(; I_bg=8)
            ni = HHNeuronInhib(; I_bg=1)
            synapse = GABA_B_Synapse()
        end
        @connections begin
            ni => ne , [weight=1000, synapse_type=GABA_B_Synapse]
        end
    end

    prob = ODEProblem(g, [], (0, 1000))
    sol = solve(prob, Vern7())
    fre_gabab = firing_rate(ne, sol; threshold=0)
    fri_gabab = firing_rate(ni, sol; threshold=0)

    @test fre_gabaa > fre_gabab
end
