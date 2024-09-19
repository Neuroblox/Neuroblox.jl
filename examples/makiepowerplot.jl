using Neuroblox
using DifferentialEquations
using CairoMakie
using DifferentialEquations

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
