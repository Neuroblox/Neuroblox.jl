using Neuroblox
using StochasticDiffEq
using Random
using Statistics
using Optimization
using OptimizationOptimJL
using ModelingToolkit: setp, getp


struct PopConfig
    name::Symbol
    blox
    firing_rate_target::Union{Nothing, Float64}
    firing_rate_weight::Float64
    freq_target::Union{Nothing, Float64}
    freq_weight::Float64
    freq_min::Union{Nothing, Float64}
    freq_max::Union{Nothing, Float64}
    freq_aggregate_state::Union{Nothing, String}
    tunable::Bool
end

function PopConfig(
    name, 
    blox; 
    firing_rate_target=nothing,
    firing_rate_weight=0.0,
    freq_target=nothing,
    freq_weight=0.0,
    freq_min=nothing,
    freq_max=nothing,
    freq_aggregate_state=nothing,
    tunable=false
)
    PopConfig(
        name, 
        blox, 
        firing_rate_target, 
        firing_rate_weight, 
        freq_target, 
        freq_weight, 
        freq_min, 
        freq_max, 
        freq_aggregate_state,
        tunable
    )
end

function get_peak_freq(powspec, freq_min, freq_max;
                        freq_ind = findall(x -> x > freq_min && x < freq_max, powspec.freq))

    if isempty(freq_ind)
        return NaN
    end
    ind = argmax(powspec.power[freq_ind])
    return powspec.freq[freq_ind][ind]
end

function get_peak_freq(powspecs::Vector, freq_min, freq_max)
    freq_ind = findall(x -> x > freq_min && x < freq_max, powspecs[1].freq)
    if isempty(freq_ind)
        return NaN
    end

    # peak_freqs = [get_peak_freq(powspec, freq_min, freq_max, freq_ind=freq_ind) for powspec in powspecs]
    # mean_peak_freq = mean(peak_freqs)
    # std_peak_freq = std(peak_freqs)

    mean_power = mean(powspec.power[freq_ind] for powspec in powspecs)
    ind = argmax(mean_power)

    return powspecs[1].freq[freq_ind][ind], NaN64
    # return mean_peak_freq, std_peak_freq
end

function get_inds(params_str::Vector{String}, pattern::Vector{String})
    # Find indices where all patterns appear
    findall(str -> all(pat -> occursin(pat, str), pattern), params_str)
end

function create_problem(size, T)
    @info "Size = $size"
    Random.seed!(123)
    N_MSN = round(Int64, 100*size)
    N_FSI = round(Int64, 50*size)
    N_GPe = round(Int64, 80*size)
    N_STN = round(Int64, 40*size)

    make_conn = Neuroblox.indegree_constrained_connection_matrix
    
    global_ns = :g

    ḡ_FSI_MSN = 0.6
    density_FSI_MSN = 0.15
    weight_FSI_MSN = ḡ_FSI_MSN / (N_FSI * density_FSI_MSN)
    conn_FSI_MSN = make_conn(density_FSI_MSN, N_FSI, N_MSN)

    ḡ_MSN_GPe = 2.5
    density_MSN_GPe = 0.33
    weight_MSN_GPe = ḡ_MSN_GPe / (N_MSN * density_MSN_GPe)
    conn_MSN_GPe = make_conn(density_MSN_GPe, N_MSN, N_GPe)

    ḡ_GPe_STN = 0.3
    density_GPe_STN = 0.05
    weight_GPe_STN = ḡ_GPe_STN / (N_GPe * density_GPe_STN)
    conn_GPe_STN = make_conn(density_GPe_STN, N_GPe, N_STN)
    
    ḡ_STN_FSI = 0.165
    density_STN_FSI = 0.1
    weight_STN_FSI = ḡ_STN_FSI / (N_STN * density_STN_FSI)
    conn_STN_FSI = make_conn(density_STN_FSI, N_STN, N_FSI)

    @named msn = Striatum_MSN_Adam(namespace=global_ns, N_inhib=N_MSN, I_bg=1.153064742988923*ones(N_MSN), σ=0.17256774881503584)
    @named fsi = Striatum_FSI_Adam(namespace=global_ns, N_inhib=N_FSI, I_bg=6.196201739395473*ones(N_FSI), σ=0.9548801242101033)
    @named gpe = GPe_Adam(namespace=global_ns, N_inhib=N_GPe, I_bg=3.272893843123162, σ=1.0959782801317943)
    @named stn = STN_Adam(namespace=global_ns, N_exci=N_STN, I_bg=2.2010777359961953*ones(N_STN), σ=2.9158528502583545)

    g = MetaDiGraph()
    add_edge!(g, fsi => msn, weight=weight_FSI_MSN, connection_matrix=conn_FSI_MSN)
    add_edge!(g, msn => gpe, weight=weight_MSN_GPe, connection_matrix=conn_MSN_GPe)
    add_edge!(g, gpe => stn, weight=weight_GPe_STN, connection_matrix=conn_GPe_STN)
    add_edge!(g, stn => fsi, weight=weight_STN_FSI, connection_matrix=conn_STN_FSI)

    @info "Creating system from graph"
    @named sys = system_from_graph(g)

    # For MSN-only
    # g = MetaDiGraph() ## defines a graph
    # add_blox!(g, msn) ## adds the defined blocks into the graph
    # @named sys = system_from_graph(g)

    tspan = (0.0, T)
    @info "Creating SDEProblem"
    prob = SDEProblem(sys, [], tspan, [])

    return prob, sys, msn, fsi, gpe, stn
end

function build_pop_param_indices(params_str::Vector{String}, populations)
    indices = NamedTuple{keys(populations)}(map(pop -> begin
        pop_str = string(pop.name)
        I_bg_inds = get_inds(params_str, ["I_bg", pop_str])
        σ_inds   = get_inds(params_str, ["σ", pop_str])
        if isempty(I_bg_inds)
            error("No I_bg parameters found for population $(pop.name). Check naming.")
        end
        if isempty(σ_inds)
            error("No σ parameters found for population $(pop.name). Check naming.")
        end
        (I_bg_ind = I_bg_inds, σ_ind = σ_inds)
    end, populations))
    return indices
end

function remake_prob!(prob, args, p)
    @unpack populations, pop_params, get_ps, set_ps! = args
    ps_new = get_ps(prob)

    offset = 1
    for (popname, popconfig) in pairs(populations)
        if popconfig.tunable
            # This population is tuned: assign parameters from p
            I_bg_inds = pop_params[popname].I_bg_ind
            σ_inds    = pop_params[popname].σ_ind

            ps_new[I_bg_inds] .= abs(p[offset])
            offset += 1
            ps_new[σ_inds] .= abs(p[offset])
            offset += 1
        end
    end

    set_ps!(prob, ps_new)
end

function loss(p, args)
    @unpack dt, saveat, solver, seed, other_diffeq_kwargs, threshold, transient, populations, prob, pop_params = args
    @unpack trajectories, ensemblealg = args
    Random.seed!(seed)
    remake_prob!(prob, args, p)

    ens_prob = EnsembleProblem(prob)
    sol = solve(ens_prob, solver, ensemblealg, trajectories=trajectories, dt=dt, saveat=saveat; other_diffeq_kwargs...)

    total_err = 0.0

    # Compute firing rates and frequencies
    for (popname, popconfig) in pairs(populations)
        if popconfig.firing_rate_target !== nothing
            fr_res = firing_rate(popconfig.blox, sol, threshold=threshold, transient=transient, scheduler=:dynamic)
            fr, fr_std = fr_res
            err_fr = (fr[1] - popconfig.firing_rate_target)^2 * popconfig.firing_rate_weight
            total_err += err_fr
            @info "[$popname] FR = $(fr[1]), FR std = $(fr_std[1]), FR Error = $err_fr"
        end

        if popconfig.freq_target !== nothing
            powspecs = powerspectrum(popconfig.blox, sol, popconfig.freq_aggregate_state, method=welch_pgram, window=hamming)

            # peak_freqs = Float64[]
            # for powspec in powspecs
            #     peak_freq = get_peak_freq(powspec, popconfig.freq_min, popconfig.freq_max)
            #     @show peak_freq
            #     push!(peak_freqs, peak_freq)
            # end
            # @show mean(peak_freqs)

            peak_freq, peak_freq_std = get_peak_freq(powspecs, popconfig.freq_min, popconfig.freq_max)
            err_freq = abs(peak_freq - popconfig.freq_target) * popconfig.freq_weight
            total_err += err_freq
            @info "[$popname] Peak freq = $peak_freq, Freq std = $peak_freq_std, Freq Error = $err_freq"
        end
    end

    return total_err
end

function run_optimization()
    prob, sys, msn, fsi, gpe, stn = create_problem(1.0, 5500.0)
    params_str = string.(tunable_parameters(sys))
    get_ps = getp(prob, tunable_parameters(sys))
    set_ps! = setp(prob, tunable_parameters(sys))

    populations = (

        ## only MSN

        # msn = PopConfig(
        #     :msn,
        #     msn,
        #     firing_rate_target=1.46,
        #     firing_rate_weight=3.0,
        #     freq_target=17.53,
        #     freq_weight=1.0,
        #     freq_min=5.0,
        #     freq_max=25.0,
        #     freq_aggregate_state="I_syn_msn",
        #     tunable=true
        # ),

        ## MSN - FSI

        # msn = PopConfig(
        #     :msn,
        #     msn,
        #     firing_rate_target=1.88,
        #     firing_rate_weight=3.0,
        #     tunable=false
        # ),
        # fsi = PopConfig(
        #     :fsi,
        #     fsi,
        #     firing_rate_target=10.66,
        #     firing_rate_weight=3.0,
        #     freq_target=58.63,
        #     freq_weight=1.0,
        #     freq_min=40.0,
        #     freq_max=90.0,
        #     freq_aggregate_state="I_syn_fsi",
        #     tunable=true
        # )

        ## full model in baseline conditions

        msn = PopConfig(
            :msn,
            msn,
            firing_rate_target=1.21,
            firing_rate_weight=60.0,
            freq_target=10,
            freq_weight=0.5,   
            freq_min=3.0,
            freq_max=20.0,
            freq_aggregate_state="I_syn_msn",
            tunable=false
        ),
        fsi = PopConfig(
            :fsi,
            fsi,
            firing_rate_target=13.00,
            firing_rate_weight=10.0,
            freq_target=61.14,
            freq_weight=1.0,
            freq_min=40.0,
            freq_max=90.0,
            freq_aggregate_state="I_syn_fsi",
            tunable=false
        ),
        gpe = PopConfig(
            :gpe,
            gpe,
            freq_target=85.0,
            freq_weight=0.5,
            freq_min=40.0,
            freq_max=90.0,
            freq_aggregate_state="V",
            tunable=true
        ),
        stn = PopConfig(
            :stn,
            stn,
            tunable=true
        ),


        ## full model PD conditions

        # msn = PopConfig(
        #     :msn,
        #     msn,
        #     firing_rate_target=4.85,
        #     firing_rate_weight=60.0,
        #     freq_target=15.80,
        #     freq_weight=1.0,   
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="I_syn_msn",
        #     tunable=true
        # ),
        # fsi = PopConfig(
        #     :fsi,
        #     fsi,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="I_syn_fsi",
        #     tunable=false
        # ),
        # gpe = PopConfig(
        #     :gpe,
        #     gpe,
        #     firing_rate_target=50.0,
        #     firing_rate_weight=1.0,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="V",
        #     tunable=false
        # ),
        # stn = PopConfig(
        #     :stn,
        #     stn,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     firing_rate_target=42.30,
        #     firing_rate_weight=1.0,
        #     tunable=false
        # ),
    );

    pop_params = build_pop_param_indices(params_str, populations)

    args = (
        dt = 0.05,
        saveat = 0.05,
        solver = RKMil(),
        seed = 1234,
        other_diffeq_kwargs = (abstol=1e-3, reltol=1e-6, maxiters=1e10),
        ensemblealg = EnsembleThreads(),
        trajectories = 3,
        threshold = -35,
        transient = 200,
        populations = populations,
        pop_params = pop_params,
        prob = prob,
        get_ps = get_ps,
        set_ps! = set_ps!
    )

    callback = function (state, l)
        @info "Iteration: $(state.iter)"
        @info "Parameters: $(state.u)"
        @info "Loss: $l"
        return false
    end

    # Starting guess
    p = [3.272893843123162, 1.0959782801317943, 2.2010777359961953, 2.9158528502583545]

    # Solve optimization
    result = solve(
        Optimization.OptimizationProblem(loss, p, args),
        Optim.NelderMead();
        callback=callback,
        maxiters=200
    )

    return result
end

result = run_optimization()