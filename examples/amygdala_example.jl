using Neuroblox
using OrdinaryDiffEq 
using CairoMakie 

global_ns = :g 
#@named ASC1 = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26);
@named LA = LateralAmygdalaBlox(N_wta=10, N_ff_inh=5, N_exci=5, density=0.002, weight=1, I_bg_ar=5, I_bg_ff_inhib=0, G_syn_ff_inhib= 3; namespace=global_ns);
@named CB = CorticalBlox(N_wta=10, N_exci=5, density=0.01, weight=1, I_bg_ar=0.1,  I_bg_ff_inhib=0.5; namespace=global_ns);

@named InfC = PulsesInput(;
    namespace=global_ns,
    pulse_amp=0, 
    pulse_switch = [1 1 1 1 0], 
    pulse_width=50, 
    t_start= 0 .+ [100 250 400 550 700]
);

@named Thal_core = Thalamus(;
    namespace=global_ns,
    N_exci=25,
    I_bg=-2.5 .+ (2.75)*rand(25),
    density=0.03,
    weight=1.3
); 

g = MetaDiGraph()
add_edge!(g, InfC => Thal_core; weight = 0.1)
add_edge!(g, Thal_core => CB; connection_rule=:gradient, weight = 1, density = 0.04)
add_edge!(g, LA => CB; connection_rule=:gradient, density=0.1, weight=1)
add_edge!(g, CB => LA; connection_rule=:gradient, density=0.1, weight=1)

#add_edge!(g, ASC1 => LA; ff_inh_num = [4, 3, 2], weight=44)
#add_edge!(g, ASC1 => CB; weight = 20)

@named sys = system_from_graph(g; graphdynamics=true);
prob = ODEProblem(sys, [], (0.0, 1000), []);
sol = solve(prob, Vern7(), saveat=0.1)
