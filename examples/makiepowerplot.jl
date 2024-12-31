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
prob = ODEProblem(neuron_net, [], (0.0, 5000), [])
sol = solve(prob, Vern7())

fss = powerspectrumplot(cb, sol)
fss = powerspectrumplot(cb, sol, show_bands = false)
fss = powerspectrumplot(cb, sol; sampling_rate=0.01, method=welch_pgram)

sol = solve(prob, Vern7(), saveat=0.01)
fss = powerspectrumplot(cb, sol)

@named msn = Striatum_MSN_Adam();
sys = structural_simplify(msn.system)
prob = SDEProblem(sys, [], (0.0, 5500), [])
sol = solve(prob, RKMil(), dt=0.05, saveat=0.01)

fig = powerspectrumplot(msn, sol)

fig = powerspectrumplot(msn, sol; state = "G",
                        method=welch_pgram,
                        window=hamming,
                        axis=(xlabel="My axis name", backgroundcolor = :purple),
                        figure = (; size=(800, 600)))
                        
fig = Figure(resolution = (1500, 600))
powerspectrumplot(fig[1,1], msn, sol; state = "G",
                        beta_start = 14,
                        axis = (; title = "Periodogram with no window"))


powerspectrumplot(fig[1,2], msn, sol; state = "G",
                        method=welch_pgram, window=hanning,
                        gamma_start = 40,
                        axis = (; title = "Welch's method with Hanning window"))

fig

powspec1 = powerspectrum(msn, sol, "G", method=welch_pgram, window=hanning)
powspec2 = powerspectrum(msn, sol, "V", method=welch_pgram, window=hanning)

powerspectrumplot(powspec1)

p = [powspec1, powspec2]
powerspectrumplot(p)
powerspectrumplot(p, show_bands=true, ylims = (-30, 20), band_labels_vertical_position="bottom", band_labels_vertical_offset = 0.01)


ens_prob = EnsembleProblem(prob)
ens_sol = solve(ens_prob, RKMil(); dt=0.05, saveat=0.05, trajectories=5)

powerspectrumplot(msn, ens_sol; state = "G",
                    ylims=(-25, 10),
                    method=welch_pgram, window=hanning,
                    axis = (; title = "Welch's method with Hanning window"))