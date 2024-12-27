using Neuroblox
using StochasticDiffEq
using Random
using Statistics
using Optimization
using OptimizationOptimJL
using ModelingToolkit: setp, getp
using DSP
using AbstractFFTs

@kwdef struct PopConfig{BLOX}
    name::Symbol
    blox::BLOX
    firing_rate_target::Union{Nothing, Float64} = nothing
    firing_rate_weight::Float64 = 1.0
    freq_target::Union{Nothing, Float64} = nothing
    freq_weight::Float64 = 1.0
    freq_min::Union{Nothing, Float64} = nothing
    freq_max::Union{Nothing, Float64} = nothing
    freq_aggregate_state::Union{Nothing, String} = nothing
    tunable::Bool = false
end

const PopIndexNT = NamedTuple{(:I_bg_ind, :σ_ind), Tuple{Vector{Int}, Vector{Int}}}

@kwdef struct Args{GetterT,SetterT,BLOXTuple,ProblemT,SolverT,DiffeqkargsT,PopT,PopparamsT}
    dt::Float64 = 0.05
    saveat::Float64 = 0.05
    solver::SolverT = RKMil()
    seed::Int = 1234
    other_diffeq_kwargs::DiffeqkargsT = (abstol=1e-3, reltol=1e-6, maxiters=1e10)
    ensemblealg::EnsembleThreads = EnsembleThreads()
    trajectories::Int = 3
    threshold::Float64 = -35
    transient::Float64 = 200
    populations::PopT
    pop_params::PopparamsT
    prob::ProblemT
    get_ps::GetterT
    set_ps!::SetterT
end


function get_peak_freq(
    powspecs,
    freq_min,
    freq_max)
    freq_ind = get_freq_inds(powspecs[1].freq, freq_min, freq_max)
    if isempty(freq_ind)
            return NaN64, NaN64
    end

    # peak_freqs = [get_peak_freq(powspec, freq_min, freq_max, freq_ind=freq_ind) for powspec in powspecs]
    # mean_peak_freq = mean(peak_freqs)
    # std_peak_freq = std(peak_freqs)

    mean_power = mean(powspec.power[freq_ind] for powspec in powspecs)
    ind = argmax(mean_power)

    return powspecs[1].freq[freq_ind][ind], NaN64
end

function get_freq_inds(freq, freq_min, freq_max)
    return findall(x -> x > freq_min && x < freq_max, freq)
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
    @named gpe = GPe_Adam(namespace=global_ns, N_inhib=N_GPe, I_bg=3.272893843123162*ones(N_GPe), σ=1.0959782801317943)
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
    prob = SDEProblem{true}(sys, [], tspan, [])

    return prob, sys, msn, fsi, gpe, stn
end

function build_pop_param_indices(params_str::Vector{String},
                                 populations::Tuple{Vararg{PopConfig}})
    idxs = map(popconfig -> begin
        pop_str = string(popconfig.name)
        I_bg_inds = get_inds(params_str, ["I_bg", pop_str])
        σ_inds    = get_inds(params_str, ["σ", pop_str])
        if isempty(I_bg_inds)
            error("No I_bg parameters found for population $(popconfig.name). Check naming.")
        end
        if isempty(σ_inds)
            error("No σ parameters found for population $(popconfig.name). Check naming.")
        end
        (I_bg_ind = I_bg_inds, σ_ind = σ_inds)
    end, populations)

    return collect(idxs)
end


function remake_prob!(prob, args, p)
    @unpack populations, pop_params, get_ps, set_ps! = args
    ps_new = get_ps(prob)

    offset = 1
    for (i, popconfig) in enumerate(populations)
        if popconfig.tunable
            I_bg_inds = pop_params[i][:I_bg_ind]
            σ_inds    = pop_params[i][:σ_ind]

            ps_new[I_bg_inds] .= abs(p[offset])
            offset += 1
            ps_new[σ_inds] .= abs(p[offset])
            offset += 1
        end
    end

    set_ps!(prob, ps_new)
    return nothing
end

function loss(p, args)
    @unpack dt, saveat, solver, seed, other_diffeq_kwargs, threshold, transient = args
    @unpack populations, prob, pop_params, trajectories, ensemblealg = args
    Random.seed!(seed)

    # Update prob in-place
    remake_prob!(prob, args, p)

    ens_prob = EnsembleProblem(prob)
    sol = solve(ens_prob, solver, ensemblealg;
                trajectories=trajectories,
                dt=dt,
                saveat=saveat,
                other_diffeq_kwargs...)

    total_err = 0.0

    # Compute firing rates and frequencies
    for popconfig in populations
        if popconfig.firing_rate_target !== nothing
            fr, fr_std = firing_rate(popconfig.blox, sol;
                                     threshold=threshold,
                                     transient=transient,
                                     scheduler=:dynamic)
            err_fr = (fr[1] - popconfig.firing_rate_target)^2 * popconfig.firing_rate_weight
            total_err += err_fr
            @info "[$(popconfig.name)] FR = $(fr[1]), FR std = $(fr_std[1]), FR Error = $err_fr"
        end

        if popconfig.freq_target !== nothing
            powspecs = powerspectrum(popconfig.blox, sol,
                                     popconfig.freq_aggregate_state;
                                     method=welch_pgram,
                                     window=hamming)
            peak_freq, peak_freq_std = get_peak_freq(powspecs, popconfig.freq_min, popconfig.freq_max)

            err_freq = abs(peak_freq - popconfig.freq_target) * popconfig.freq_weight
            total_err += err_freq
            @info "[$(popconfig.name)] Peak freq = $peak_freq, Freq std = $peak_freq_std, Freq Error = $err_freq"
        end
    end

    return total_err
end

function run_optimization()
    prob, sys, msn, fsi, gpe, stn = create_problem(0.1, 5500.0)
    params_str = string.(tunable_parameters(sys))
    get_ps = getp(prob, tunable_parameters(sys))
    set_ps! = setp(prob, tunable_parameters(sys))

    populations = (

        ## only MSN

        # PopConfig(
        #     name=:msn,
        #     blox=msn,
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

        # PopConfig(
        #     name=:msn,
        #     blox=msn,
        #     firing_rate_target=1.88,
        #     firing_rate_weight=3.0,
        #     tunable=false
        # ),
        # PopConfig(
        #     name=:fsi,
        #     blox=fsi,
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

        PopConfig(
            name=:msn,
            blox=msn,
            firing_rate_target=1.21,
            firing_rate_weight=60.0,
            freq_target=10.0,
            freq_weight=0.5,   
            freq_min=3.0,
            freq_max=20.0,
            freq_aggregate_state="I_syn_msn",
            tunable=false
        ),
        PopConfig(
            name=:fsi,
            blox=fsi,
            firing_rate_target=13.00,
            firing_rate_weight=10.0,
            freq_target=61.14,
            freq_weight=1.0,
            freq_min=40.0,
            freq_max=90.0,
            freq_aggregate_state="I_syn_fsi",
            tunable=false
        ),
        PopConfig(
            name=:gpe,
            blox=gpe,
            freq_target=85.0,
            freq_weight=0.5,
            freq_min=40.0,
            freq_max=90.0,
            freq_aggregate_state="V",
            tunable=true
        ),
        PopConfig(
            name=:stn,
            blox=stn,
            tunable=true
        ),


        ## full model PD conditions

        # PopConfig(
        #     name=:msn,
        #     blox=msn,
        #     firing_rate_target=4.85,
        #     firing_rate_weight=60.0,
        #     freq_target=15.80,
        #     freq_weight=1.0,   
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="I_syn_msn",
        #     tunable=true
        # ),
        # PopConfig(
        #     name=:fsi,
        #     blox=fsi,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="I_syn_fsi",
        #     tunable=false
        # ),
        # PopConfig(
        #     name=:gpe,
        #     blox=gpe,
        #     firing_rate_target=50.0,
        #     firing_rate_weight=1.0,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_max=50.0,
        #     freq_aggregate_state="V",
        #     tunable=false
        # ),
        # PopConfig(
        #     name=:stn,
        #     blox=stn,
        #     freq_target=15.80,
        #     freq_weight=1.0,
        #     freq_min=3.0,
        #     freq_aggregate_state="V",
        #     freq_max=50.0,
        #     firing_rate_target=42.30,
        #     firing_rate_weight=1.0,
        #     tunable=false
        # ),
    );

    pop_params = build_pop_param_indices(params_str, populations)

    other_diffeq_kwargs = (abstol=1e-3, reltol=1e-6, maxiters=1e10)
    solver =  RKMil()
    args = Args{typeof(get_ps), typeof(set_ps!), typeof(populations), typeof(prob), typeof(solver), typeof(other_diffeq_kwargs), typeof(populations), typeof(pop_params)}(
        dt = 0.05,
        saveat = 0.05,
        solver = solver,
        seed = 1234,
        other_diffeq_kwargs = other_diffeq_kwargs,
        ensemblealg = EnsembleThreads(),
        trajectories = 3,
        threshold = -35.0,
        transient = 200.0,
        populations = populations,  # the tuple
        pop_params = pop_params,    # the vector of indices
        prob = prob,                # the SDEProblem
        get_ps = get_ps,
        set_ps! = set_ps!
    )

    callback = function (state, l)
        @info "Iteration: $(state.iter)"
        @info "Parameters: $(state.u)"
        @info "Loss: $l"
        println("\n")
        return false
    end

    # Starting guess
    u = [3.272893843123162, 1.0959782801317943, 2.2010777359961953, 2.9158528502583545]

    # Solve optimization
    result = solve(
        Optimization.OptimizationProblem(loss, u, args),
        Optim.NelderMead();
        callback=callback,
        maxiters=200
    )

    return result
end

result = run_optimization()