using Neuroblox
using OrdinaryDiffEq 
using CairoMakie 
using Optimization
using OptimizationOptimJL

using Optim

function objective(x, data)
    I_bg_CB, I_bg_LA, pulse_amp, dens_CB_LA, dens_LA_CB, dens_LA = x

    global_ns = :g
    N_LA_clusters = 5

    LA_clusters = map(Base.OneTo(N_LA_clusters)) do i
        LateralAmygdalaCluster(;
            name=Symbol("LA_$i"), 
            namespace=global_ns, 
            N_wta=10, 
            N_exci=5, 
            density=dens_LA, 
            weight=1, 
            I_bg_ar=I_bg_LA, 
            I_bg_ff_inhib=0.2, 
        )
    end
    
    @named CB = CorticalBlox(; 
        namespace=global_ns, 
        N_wta=10, 
        N_exci=5, 
        density=0.05, 
        weight=1, 
        I_bg_ar=I_bg_CB,
        I_bg_ff_inhib=0.2
    );

    @named Thal_core = Thalamus(;
        namespace=global_ns,
        N_exci=25,
        I_bg=-2.5 .+ (2.75)*rand(25),
        density=0.03,
        weight=1.3
    ); 

    @named InfC = PulsesInput(;
        namespace=global_ns, 
        pulse_width=1000, 
        pulse_amp=pulse_amp,
        t_start = [250]
    );

    g = MetaDiGraph()
    add_edge!(g, InfC => Thal_core; weight = 1)
    add_edge!(g, Thal_core => CB; weight = 1, density = 0.04)

    for i in Base.OneTo(N_LA_clusters)
        add_edge!(g, LA_clusters[i] => CB; connection_rule=:gradient, density=dens_LA_CB, weight=1)
        add_edge!(g, CB => LA_clusters[i]; connection_rule=:gradient, density=dens_CB_LA, weight=1)
    end
    
    @named sys = system_from_graph(g; graphdynamics=true);
    prob = ODEProblem(sys, [], (0.0, 2000), []);
    sol = solve(prob, TRBDF2(); saveat=0.1, abstol=1e-2, reltol=1e-2)

    n_exci = mapreduce(x -> Neuroblox.get_exci_neurons(x), vcat, LA_clusters)
    fr = firing_rate(n_exci, sol; threshold=-10, win_size=100)
    
    SciMLBase.successful_retcode(sol) || return Inf
    return sum((fr .- data).^2)
end

fr_data = [0.75, 0.7, 0.9, 1.2, 0.8, 1.3, 1.6, 1.1, 1.2, 1.1, 0.8, 0.8, 0.8, 0.8, 1.1, 1.0, 0.6, 0.5, 0.6, 0.75] 

p0 = [1, 1, 1, 0.05, 0.05, 0.05]
lb = [-5, -5, 0, 0, 0, 0]
ub = [5, 5, 5, 1, 1, 1]

using Logging
Logging.disable_logging(Logging.Info)

sol = optimize(x -> objective(x, fr_data), p0, LBFGS())

#=
obj = OptimizationFunction(
    (p, hyperp) -> objective(p, fr_data), 
    Optimization.AutoFiniteDiff()
)

prob = OptimizationProblem(obj, p0; lb, ub)

alg = BFGS()
sol = solve(prob, alg)
=#
