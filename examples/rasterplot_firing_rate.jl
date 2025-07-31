using Neuroblox
using DifferentialEquations
using CairoMakie # due to a bug in CairoMakie, we need to use CairoMakie@0.11


@named msn = Striatum_MSN_Adam(I_bg = 1.172*ones(100), σ = 0.11);
sys = structural_simplify(msn.system)
prob = SDEProblem(sys, [], (0.0, 5500.0), [])
sol = solve(prob, RKMil(); dt=0.05, saveat=0.05)

rasterplot(msn, sol; threshold=-50)

spikes = detect_spikes(msn, sol; threshold=-55)
t, fr = mean_firing_rate(spikes, sol)




global_ns = :g # global namespace
@named n1 = LIFExciCircuitBlox(; N_neurons= 5, namespace = global_ns);
@named n2 = LIFInhCircuitBlox(; N_neurons= 5, namespace = global_ns);

neurons = [n1, n2]

g = MetaDiGraph()
add_blox!.(Ref(g), neurons)

add_edge!(g, 2, 1, Dict(:weight=> 5))

@named sys = system_from_graph(g)
sys_simpl = structural_simplify(sys)
prob = ODEProblem(sys_simpl, [], (0, 200.0))
sol = solve(prob, Tsit5())

rasterplot(n1, sol) # error