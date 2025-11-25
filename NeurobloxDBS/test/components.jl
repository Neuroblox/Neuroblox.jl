using NeurobloxDBS
using StochasticDiffEq
using DelayDiffEq
using DataFrames
using Test
using Distributions
using Statistics
using LinearAlgebra
using Random

@testset "DBS circuit firing rates" begin
    @testset "Striatum_MSN_Adam" begin
        Random.seed!(1234)
        @named msn = Striatum_MSN_Adam(N_inhib = 10)
        sys = structural_simplify(get_system(msn))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(msn, ens_sol, threshold=-35, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 5.74, atol = 0.49)
    end
    @testset "Striatum_FSI_Adam" begin
        Random.seed!(1234)
        @named fsi = Striatum_FSI_Adam(N_inhib = 10)
        sys = structural_simplify(get_system(fsi))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(fsi, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 12.02, atol = 0.1)
    end
    @testset "GPe_Adam" begin
        Random.seed!(1234)
        @named gpe = GPe_Adam(N_inhib = 10, density = 0.5, weight = 0.5)
        sys = structural_simplify(get_system(gpe))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(gpe, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 32.46, atol = 0.18)
    end
    @testset "STN_Adam" begin
        Random.seed!(1234)
        @named stn = STN_Adam(N_exci = 10, density = 0.5, weight = 0.5)
        sys = structural_simplify(get_system(stn))
        prob = SDEProblem(sys, [], (0.0, 6500.0), [])
        ens_sol = solve(EnsembleProblem(prob), RKMil(); dt=0.05, saveat=0.05, trajectories=5, abstol=1e-2, reltol=1e-2);
        mean_fr, std_fr = firing_rate(stn, ens_sol, threshold=-25, transient=1000, scheduler=:dynamic)
        @test isapprox(mean_fr[1], 292.63, atol = 4.79)
    end
end
