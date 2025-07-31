using Neuroblox
using GraphDynamics
using OrdinaryDiffEq ## to build the ODE problem and solve it, gain access to multiple solvers from this (AFAIK this is lighter than using DifferentialEquations)
using Distributions ## for statistical distributions 
#using CairoMakie

function run_dm(N)
    ## Describe what the local variables you define are for
    global_ns = :g ## global name for the circuit. All components should be inside this namespace.
    tspan = (0, 1000) ## Simulation time span [ms]
    spike_rate = 2.4 ## spikes / ms

    f = 0.15 ## ratio of selective excitatory to non-selective excitatory neurons
    f_inh = 0.2
    N_E = Int(N * (1 - f_inh)) 
    N_I = Int(ceil(N * f_inh)) ## total number of inhibitory neurons
    N_E_selective = Int(ceil(f * N_E)) ## number of selective excitatory neurons
    N_E_nonselective = N_E - 2 * N_E_selective ## number of non-selective excitatory neurons

    w₊ = 1.7 
    w₋ = 1 - f * (w₊ - 1) / (1 - f)

    ## Use scaling factors for conductance parameters so that our abbreviated model 
    ## can exhibit the same competition behavior between the two selective excitatory populations
    ## as the larger model in Wang 2002 does.
    exci_scaling_factor = 1600 / N_E
    inh_scaling_factor = 400 / N_I

    coherence = 0 # random dot motion coherence [%]
    dt_spike_rate = 50 # update interval for the stimulus spike rate [ms]
    μ_0 = 40e-3 # mean stimulus spike rate [spikes / ms]
    ρ_A = ρ_B = μ_0 / 100
    μ_A = μ_0 + ρ_A * coherence
    μ_B = μ_0 + ρ_B * coherence 
    σ = 4e-3 # standard deviation of stimulus spike rate [spikes / ms]

    spike_rate_A = Normal(μ_A, σ) => dt_spike_rate # spike rate distribution for selective population A
    spike_rate_B = Normal(μ_B, σ) => dt_spike_rate # spike rate distribution for selective population B

    # Blox definitions
    @named background_input = PoissonSpikeTrain(spike_rate, tspan; namespace = global_ns, N_trains=1);
    @named stim_A = PoissonSpikeTrain(spike_rate_A, tspan; namespace = global_ns);
    @named stim_B = PoissonSpikeTrain(spike_rate_B, tspan; namespace = global_ns);
    @named n_A = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor);
    @named n_B = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) ;
    @named n_ns = LIFExciCircuitBlox(; namespace = global_ns, N_neurons = N_E_nonselective, weight = 1.0, exci_scaling_factor, inh_scaling_factor);
    @named n_inh = LIFInhCircuitBlox(; namespace = global_ns, N_neurons = N_I, weight = 1.0, exci_scaling_factor, inh_scaling_factor);

    ## This is a convenience step so that we can later add edges to the graph using the Blox names.
    ## (We should replace add_edge! with a nicer interface to avoid this eventually)
    g = MetaDiGraph()
    add_edge!(g, background_input => n_A; weight = 1);
    add_edge!(g, background_input => n_B; weight = 1);
    add_edge!(g, background_input => n_ns; weight = 1);
    add_edge!(g, background_input => n_inh; weight = 1);

    add_edge!(g, stim_A => n_A; weight = 1);
    add_edge!(g, stim_B => n_B; weight = 1);

    add_edge!(g, n_A => n_B; weight = w₋);
    add_edge!(g, n_A => n_ns; weight = 1);
    add_edge!(g, n_A => n_inh; weight = 1);

    add_edge!(g, n_B => n_A; weight = w₋);
    add_edge!(g, n_B => n_ns; weight = 1);
    add_edge!(g, n_B => n_inh; weight = 1);

    add_edge!(g, n_ns => n_A; weight = w₋);
    add_edge!(g, n_ns => n_B; weight = w₋);
    add_edge!(g, n_ns => n_inh; weight = 1);

    add_edge!(g, n_inh => n_A; weight = 1);
    add_edge!(g, n_inh => n_B; weight = 1);
    add_edge!(g, n_inh => n_ns; weight = 1);

    ## Build the ODE system from the model graph
    sys = system_from_graph(g; name=global_ns, graphdynamics = true);
    ## Build an ODE Problem object out of the system
    prob = ODEProblem(sys, [], tspan);
    #sol = solve(prob, TRBDF2(autodiff=false))
    sol = solve(prob, Euler(); dt = 0.01);
    
    return sol
end

#rasterplot(n_inh, sol; color=:red)
#rasterplot(n_A, sol)
#rasterplot(n_B, sol)
