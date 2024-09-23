using Neuroblox
using DifferentialEquations
using CairoMakie # due to a bug in CairoMakie, we need to use CairoMakie@0.11

global_ns = :g 
@named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
@named cb = CorticalBlox(N_wta=2, N_exci=2, namespace=global_ns, density=0.1, weight=1)
assembly = [LC, cb]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g,1,2, :weight, 44)
neuron_net = system_from_graph(g; name=global_ns)
prob = ODEProblem(structural_simplify(neuron_net), [], (0.0, 600), [])

sol = solve(prob, Vern7())
fss = band_power_meanfield(cb, sol)
fss = band_power_meanfield(cb, sol; sampling_rate=0.01)

sol = solve(prob, Vern7(), saveat=0.05)
fss = band_power_meanfield(cb, sol)



@named msn = Striatum_MSN_Adam(I_bg = 1.17*ones(100), σ = 0.11);
sys = structural_simplify(msn.odesystem)
prob = SDEProblem(sys, [], (0.0, 5500), [])
sol = solve(prob, RKMil(), dt=0.05, saveat=0.05)

fss = band_power_meanfield(msn, sol, ylims=(1e-5, 10),
                            alpha_label_position = (8.5, 4.0),
                            beta_label_position = (22, 4.0),
                            gamma_label_position = (60, 4.0))

fig = band_power_meanfield(msn, sol, "G",
                            powerspectrum_kwargs = (; method=Neuroblox.welch_pgram),
                            ylims=(1e-5, 10), show_bands = false,
                            axis=(xlabel="My axis name", backgroundcolor = :purple),
                            figure = (; size=(800, 300)))


fig = band_power_meanfield(msn, sol, "G",
                            powerspectrum_kwargs = (;method=Neuroblox.welch_pgram, window=Neuroblox.hanning),
                            ylims=(1e-4, 2e-1), xlims=(2, 100),
                            alpha_start = 5,
                            alpha_label_position = (8.5, 1.2e-1),
                            beta_label_position = (22, 1.2e-1),
                            gamma_label_position = (60, 1.2e-1))