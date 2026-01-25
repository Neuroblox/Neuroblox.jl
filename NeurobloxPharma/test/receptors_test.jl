using GraphDynamics
using NeurobloxBase
using NeurobloxPharma
using OrdinaryDiffEqTsit5
using SciMLBase: ReturnCode
using CairoMakie, ReferenceTests
using Test

include("receptors_support.jl")

function plot_trn_tonic_spikes()
    _, t, v = trn_tonic_voltage_trace()
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="time (ms)", ylabel="V_m (mV)",
        title="TRN tonic (g_KCa=0, g_CAN=0.05)")
    lines!(ax, t, v; label="I_app=6.0")
    fig[1, 2] = Legend(fig, ax, framevisible=false)
    fig
end

function plot_baxter_tonic_spikes()
    _, t, v = baxter_tonic_voltage_trace()
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="time (ms)", ylabel="V (mV)",
        title="Baxter tonic (KCaS=0)")
    lines!(ax, t, v; label="I_stim=1.0")
    fig[1, 2] = Legend(fig, ax, framevisible=false)
    fig
end

@testset "MsnD1Receptor modulates NMDA current" begin
    sol0, _, phi0 = d1_phi_trace(da=0.0)
    sol1, _, phi1 = d1_phi_trace(da=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test phi1[end] > phi0[end] + 1e-3

    sol_nmda, _, _ = nmda_trace()
    @test sol_nmda.retcode == ReturnCode.Success

    @named d1 = MsnD1Receptor()
    beta1 = GD.to_subsystem(d1).β1
    i_base = nmda_current(1.0; v=0.0)
    i_d1 = nmda_current(1.0 + beta1 * phi1[end]; v=0.0)
    @test i_base != 0.0
    @test abs(i_d1) > abs(i_base)
end

@testset "MsnD2Receptor modulates AMPA current" begin
    sol0, _, phi0 = d2_phi_trace(da=0.0)
    sol1, _, phi1 = d2_phi_trace(da=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test phi1[end] > phi0[end] + 1e-3

    @named d2 = MsnD2Receptor()
    beta2 = GD.to_subsystem(d2).β2
    m_ampa = 1.0 - beta2 * phi1[end]
    sol_base, i_base = ampa_current(1.0)
    sol_d2, i_d2 = ampa_current(m_ampa)
    @test sol_base.retcode == ReturnCode.Success
    @test sol_d2.retcode == ReturnCode.Success
    @test abs(i_d2) < abs(i_base)
end

@testset "HTR5 mode flags" begin
    sol, _, _ = htr5_trace(mode=0.0)
    @test sol.retcode == ReturnCode.Success
    f0 = htr5_flags(0.0)
    f3 = htr5_flags(3.0)
    @test f0.PKA == 0.0
    @test f0.PKC == 0.0
    @test f3.PKA == 1.0
    @test f3.PKC == 1.0
end

@testset "MuscarinicR activation" begin
    sol0, _, phi0 = muscarinic_phi_trace(m=0.0)
    sol1, _, phi1 = muscarinic_phi_trace(m=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test phi1[end] > phi0[end] + 1e-3
end

@testset "MuscarinicR -> MuscarinicNeuron V" begin
    sol0, t0, v0 = muscarinic_pair_voltage_trace(m=0.0, g_ncm=1.0)
    sol1, t1, v1 = muscarinic_pair_voltage_trace(m=1.0, g_ncm=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "MuscarinicNeuron V response" begin
    sol0, t0, v0 = muscarinic_neuron_voltage_trace(i_app=0.0)
    sol1, t1, v1 = muscarinic_neuron_voltage_trace(i_app=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "Alpha7ERnAChR -> TRNNeuron V" begin
    sol0, t0, v0 = alpha7_trn_voltage_trace(ach=0.0)
    sol1, t1, v1 = alpha7_trn_voltage_trace(ach=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "CaTRPM4R -> TRNNeuron V" begin
    sol0, t0, v0 = catrpm4_trn_voltage_trace(cch=0.0)
    sol1, t1, v1 = catrpm4_trn_voltage_trace(cch=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "TRNNeuron V response" begin
    sol0, t0, v0 = trn_voltage_trace(i_app=0.0)
    sol1, t1, v1 = trn_voltage_trace(i_app=1.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "TRNNeuron tonic spikes" begin
    sol, t, v = trn_tonic_voltage_trace()
    @test sol.retcode == ReturnCode.Success
    @test t[end] >= V_TRACE_TSPAN[2]
    @test count_spikes(t, v; thresh=0.0, refractory_ms=5.0) >= 3
    fig = plot_trn_tonic_spikes()
    @test_reference "plots/TRNNeuron_tonic_spikes.png" fig by = psnr_equality(25)
end

@testset "BaxterSensoryNeuron V response" begin
    sol0, t0, v0 = baxter_voltage_trace(I_stim=0.0)
    sol1, t1, v1 = baxter_voltage_trace(I_stim=0.5)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1e-3
end

@testset "BaxterSensoryNeuron tonic spikes" begin
    sol, t, v = baxter_tonic_voltage_trace()
    @test sol.retcode == ReturnCode.Success
    @test t[end] >= V_TRACE_TSPAN[2]
    @test count_spikes(t, v; thresh=0.0, refractory_ms=5.0) >= 3
    fig = plot_baxter_tonic_spikes()
    @test_reference "plots/BaxterSensoryNeuron_tonic_spikes.png" fig by = psnr_equality(25)
end

# Beta2nAChR tests
@testset "Beta2nAChR ACh activation" begin
    sol0, _, act0 = beta2nachr_activation_trace(inp_ACh=0.0, inp_Nic=0.0)
    sol1, _, act1 = beta2nachr_activation_trace(inp_ACh=10.0, inp_Nic=0.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    # ACh input should activate the receptor
    @test act1[end] > act0[end] + 0.1
end

@testset "Beta2nAChR nicotine desensitization" begin
    sol0, _, des0 = beta2nachr_des_trace(inp_ACh=0.0, inp_Nic=0.0)
    sol1, _, des1 = beta2nachr_des_trace(inp_ACh=0.0, inp_Nic=0.5)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    # Nicotine should cause desensitization
    @test des1[end] > des0[end] + 0.1
    @test des0[end] < 0.1
end

@testset "VTADANeuron V response" begin
    sol0, t0, v0 = vtada_voltage_trace(i_app=0.0)
    sol1, t1, v1 = vtada_voltage_trace(i_app=10.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1.0
end

@testset "VTAGABANeuron V response" begin
    sol0, t0, v0 = vtagaba_voltage_trace(i_app=0.0)
    sol1, t1, v1 = vtagaba_voltage_trace(i_app=10.0)
    @test sol0.retcode == ReturnCode.Success
    @test sol1.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]
    @test max_trace_delta(v0, v1) > 1.0
end

@testset "Beta2nAChR -> VTADANeuron V" begin
    sol0, t0, v0 = beta2nachr_neuron_voltage_trace(inp_ACh=0.0, inp_Nic=0.0, g_ACh=0.5)
    @test sol0.retcode == ReturnCode.Success
    @test t0[end] >= V_TRACE_TSPAN[2]

    # Test depolarization with ACh
    sol1, t1, v1 = beta2nachr_neuron_voltage_trace(inp_ACh=10.0, inp_Nic=0.0, g_ACh=0.5)
    @test sol1.retcode == ReturnCode.Success
    @test sum(v1) / length(v1) > sum(v0) / length(v0) + 1.0
end

function plot_vtada_tonic_spikes()
    _, t, v = vtada_tonic_voltage_trace()
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="time (ms)", ylabel="V (mV)",
        title="VTADANeuron tonic (I_app=10.0)")
    lines!(ax, t, v; label="I_app=10.0")
    fig[1, 2] = Legend(fig, ax, framevisible=false)
    fig
end

function plot_vtagaba_tonic_spikes()
    _, t, v = vtagaba_tonic_voltage_trace()
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="time (ms)", ylabel="V (mV)",
        title="VTAGABANeuron tonic (I_app=10.0)")
    lines!(ax, t, v; label="I_app=10.0")
    fig[1, 2] = Legend(fig, ax, framevisible=false)
    fig
end

@testset "VTADANeuron tonic spikes" begin
    sol, t, v = vtada_tonic_voltage_trace()
    @test sol.retcode == ReturnCode.Success
    @test t[end] >= V_TRACE_TSPAN[2]
    @test count_spikes(t, v; thresh=0.0, refractory_ms=5.0) >= 3
    fig = plot_vtada_tonic_spikes()
    @test_reference "plots/VTADANeuron_tonic_spikes.png" fig by = psnr_equality(25)
end

@testset "VTAGABANeuron tonic spikes" begin
    sol, t, v = vtagaba_tonic_voltage_trace()
    @test sol.retcode == ReturnCode.Success
    @test t[end] >= V_TRACE_TSPAN[2]
    @test count_spikes(t, v; thresh=0.0, refractory_ms=5.0) >= 3
    fig = plot_vtagaba_tonic_spikes()
    @test_reference "plots/VTAGABANeuron_tonic_spikes.png" fig by = psnr_equality(25)
end

