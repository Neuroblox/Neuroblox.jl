using Neuroblox
using OrdinaryDiffEq ## to build the ODE problem and solve it, gain access to multiple solvers from this
using CairoMakie ## for customized plotting recipies for blox
using CSV ## to read data from CSV files
using DataFrames ## to format the data into DataFrames

global_ns = :g 
@named ASC1 = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26);
@named ASC2 = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26);
@named ASC3 = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26);
@named LA = LateralAmygdalaBlox(N_wta=10, N_ff_inh=5, N_exci=5, density=0.002, weight=1, I_bg_ar=5, I_bg_ff_inhib=0, G_syn_ff_inhib= 3; namespace=global_ns);
@named CB = CorticalBlox(N_wta=10, N_ff_inh=5, N_exci=5, density=0.01, weight=1, I_bg_ar=0.1,  I_bg_ff_inhib=0.5; namespace=global_ns);
nn1 = HHNeuronInhibBlox(name=Symbol("nrn1"), E_syn=-70, G_syn = 5, I_bg=1.0, namespace=global_ns);
nn2 = HHNeuronInhibBlox(name=Symbol("nrn2"), E_syn=-70, G_syn = 5, I_bg=0, namespace=global_ns);
nn3 = HHNeuronInhibBlox(name=Symbol("nrn3"), E_syn=-70, G_syn = 5, I_bg=0, namespace=global_ns);
nn4 = HHNeuronInhibBlox(name=Symbol("nrn4"), E_syn=-70, G_syn = 5, I_bg=0, namespace=global_ns);
nn5 = HHNeuronInhibBlox(name=Symbol("nrn5"), E_syn=-70, G_syn = 5, I_bg=0, namespace=global_ns);

w_LA_to_CB = zeros(Float32, (250, 50))
for i in 1:250
    idx1 = Int8(ceil(i/50))
    for j in 1:50
        indx2 = Int8(ceil(j/10))
        if idx1 == indx2
            w_LA_to_CB[i,j] = 1.0
        else
            w_LA_to_CB[i,j] = 0.0
        end
    end
end

w_CB_to_LA = zeros(Float32, (50, 250))
for i in 1:50
    idx1 = Int8(ceil(i/10))
    for j in 1:250
        indx2 = Int8(ceil(j/50))
        if idx1 == indx2
            w_CB_to_LA[i,j] = 1.0
        else
            w_CB_to_LA[i,j] = 0.0
        end
    end
end

g = MetaDiGraph()
add_edge!(g, ASC1 => LA, ff_inh_num = 4, weight=44)
add_edge!(g, ASC2 => LA, ff_inh_num = 3, weight=44)
add_edge!(g, ASC3 => LA, ff_inh_num = 2, weight=44)
#add_edge!(g, LA=>CB, weightmatrix = w_LA_to_CB)
#add_edge!(g, CB=>LA, weightmatrix = w_CB_to_LA)
add_edge!(g, nn1 => LA, ff_inh_num = 4, weight = 0.5)
#add_edge!(g, nn2 => LA, ff_inh_num = 3, weight = 0.5)
#add_edge!(g, nn3 => LA, ff_inh_num = 2, weight = 0.5)
#add_edge!(g, nn4 => LA, ff_inh_num = 1, weight = 0.5)
#add_edge!(g, nn5 => LA, ff_inh_num = 0, weight = 0.5)
@named sys = system_from_graph(g; graphdynamics=true);



prob = ODEProblem(sys, [], (0.0, 1000), []);
sol = solve(prob, Vern7(), saveat=0.1)
#stackplot(neuron_set, sol)
neuron_set_LA = get_neurons(LA)
#neuron_set_CB = get_neurons(CB)
neuron_set_exci_LA = get_exci_neurons(LA)
#neuron_set_exci_CB = get_exci_neurons(CB)
neuron_set_inhib_LA = get_inh_neurons(LA)
#neuron_set_inhib_CB = get_inh_neurons(CB)

s_ASC_a = state_timeseries(ASC1, sol, "aₑ")
s_ASC_b = state_timeseries(ASC1, sol, "bₑ")
v_LA1 = voltage_timeseries(neuron_set_LA[301], sol)
v_LA2 = voltage_timeseries(neuron_set_LA[302], sol)
v_CB = voltage_timeseries(neuron_set_CB[61], sol)
v_CB_p1 = voltage_timeseries(neuron_set_CB[3], sol)
v_CB_p2 = voltage_timeseries(neuron_set_CB[10], sol)
iv1 = voltage_timeseries(nn1, sol)
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
cl = get_neuron_color(neuron_set_LA[301])
l1 = lines!(ax, sol.t, v_LA1, color=:red, label="LA1")
l2 = lines!(ax, sol.t, v_LA2, color=:orange, label="LA2")
l3 = lines!(ax, sol.t, v_CB, color=:red, label="CB")
l4 = lines!(ax, sol.t, iv1, color=:black, label="nn1")
l5 = lines!(ax, sol.t, v_CB_p1, color=:green, label="CB_p1")
l6 = lines!(ax, sol.t, v_CB_p2, color=:blue, label="CB_p2")
Legend(fig[1, 2], [l1, l2, l3, l4, l5, l6], ["LA1", "LA2", "CB", "nn1", "CB_p1", "CB_p2"])
fig
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
cl = get_neuron_color(neuron_set[301])
lines!(ax, sol.t, v1, color=cl)
fig

stackplot(neuron_set_LA[1:60], sol)





add_edge!(g, nn1 => nn2, weight = 0.5)
add_edge!(g, nn2 => nn3, weight = 0.5)
add_edge!(g, nn3 => nn4, weight = 0.2)
add_edge!(g, nn4 => nn5, weight = 0.4)
add_edge!(g, nn5 => nn1, weight = 0.5)


@named sys = system_from_graph(g)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
stackplot(neuron_set, sol)

v1 = voltage_timeseries(neuron_set[301], sol)
v2 = voltage_timeseries(neuron_set[302], sol)
v3 = voltage_timeseries(neuron_set[303], sol)
v4 = voltage_timeseries(neuron_set[304], sol)
v5 = voltage_timeseries(neuron_set[305], sol)
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
cl1 = get_neuron_color(neuron_set[301]) #specify color based on neuron type (excitatory/inhibitory)
lines!(ax, sol.t, v1, color=cl)
lines!(ax, sol.t, v2, color=cl)
lines!(ax, sol.t, v3, color=cl)
lines!(ax, sol.t, v4, color=cl)
lines!(ax, sol.t, v5, color=cl)
fig


iv1 = voltage_timeseries(nn1, sol)
iv2 = voltage_timeseries(nn2, sol)
iv3 = voltage_timeseries(nn3, sol)
iv4 = voltage_timeseries(nn4, sol)
iv5 = voltage_timeseries(nn5, sol)
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
cl = get_neuron_color(nn1) #specify color based on neuron type (excitatory/inhibitory)
lines!(ax, sol.t, iv1, color=cl)
lines!(ax, sol.t, iv2, color=cl)
lines!(ax, sol.t, iv3, color=cl)
lines!(ax, sol.t, iv4, color=cl)
lines!(ax, sol.t, iv5, color=cl)
fig