using Neuroblox
using StochasticDiffEq
using Random
using Statistics
using Optimization
using OptimizationOptimJL
using ModelingToolkit: setp, getp
using Unrolled

# Abstract type for all metrics
abstract type AbstractPopulationMetric end

###########
# Metrics #
###########
struct FiringRateMetric{T} <: AbstractPopulationMetric
    target::T
    weight::T
    threshold::T
    transient::T
end
struct FrequencyMetric{T} <: AbstractPopulationMetric
    target::T
    weight::T
    min_freq::T
    max_freq::T
    aggregate_state::String
end

#######################
# Metric Computations #
#######################

function compute_metric(m::FiringRateMetric, pop, sol; logging::Bool=false)
    fr, fr_std = firing_rate(pop.blox, sol;
                             threshold=m.threshold,
                             transient=m.transient,
                             scheduler=:dynamic)
    val = (fr[1] - m.target)^2 * m.weight

    if logging
        @info "[$(pop.name)] Firing Rate = $(fr[1]) ± $(fr_std[1]) " *
              " (target=$(m.target)), metric error = $val"
    end
    return val
end

function compute_metric(m::FrequencyMetric, pop, sol; logging::Bool=false)
    powspecs = powerspectrum(pop.blox, sol, m.aggregate_state;
                             method=welch_pgram,
                             window=hamming)
    peak_freq, peak_freq_std = get_peak_freq(powspecs, m.min_freq, m.max_freq)
    val = abs(peak_freq - m.target) * m.weight

    if logging
        @info "[$(pop.name)] Peak Frequency = $peak_freq ± $peak_freq_std " *
              " in [$(m.min_freq), $(m.max_freq)] (target=$(m.target)), metric error = $val"
    end
    return val
end

####################
# Helper Functions #
####################

function get_peak_freq(powspecs, freq_min, freq_max)
    freq_ind = get_freq_inds(powspecs[1].freq, freq_min, freq_max)
    if isempty(freq_ind)
        return NaN64, NaN64
    end

    # Average the power across the different trajectories
    mean_power = mean(powspec.power[freq_ind] for powspec in powspecs)
    ind = argmax(mean_power)

    # alternative method
    # peak_freqs = [get_peak_freq(powspec, freq_min, freq_max, freq_ind=freq_ind) for powspec in powspecs]
    # mean_peak_freq = mean(peak_freqs)
    # std_peak_freq = std(peak_freqs)

    return powspecs[1].freq[freq_ind][ind], NaN64
end

function get_freq_inds(freq, freq_min, freq_max)
    return findall(x -> x > freq_min && x < freq_max, freq)
end

struct TuningSpec
    param_map::Dict{String, Vector{Int}}
end

# Find indexes of the parameters to be tuned
function build_tuning_spec(prob, pop_name::String, param_names::Vector{String})
    paramlist = string.(tunable_parameters(prob.f.sys)) 
    param_map = Dict{String,Vector{Int}}()
    for pname in param_names
        inds = findall(str -> occursin(pname, str) && occursin(pop_name, str),
                       paramlist)
        param_map[pname] = inds
    end
    return TuningSpec(param_map)
end

################
#  Populations #
################

struct Population{B,N,MT<:NTuple{N,AbstractPopulationMetric}}
    name::Symbol
    blox::B
    metrics::MT
    tuning::TuningSpec
    tunable::Bool
end

"""
    compute_metrics(pop, sol; logging=false)

Sum the contributions of all metrics in `pop.metrics`, optionally logging.
"""
function compute_metrics(pop::Population, sol; logging::Bool=false)
    total = zero(eltype(sol))
    for m in pop.metrics
        total += compute_metric(m, pop, sol; logging=logging)
    end
    return total
end

function Population(
    name,
    blox;
    frm=nothing,
    freqm=nothing,
    tuning_params=String[],
    prob=nothing,
    tunable::Bool=false
)
    mt = ()
    if frm !== nothing
        fr_target, fr_weight, fr_threshold, fr_transient = frm
        mt = (FiringRateMetric(fr_target, fr_weight, fr_threshold, fr_transient),)
    end
    if freqm !== nothing
        freq_target, freq_weight, fmin, fmax, agg = freqm
        freq = FrequencyMetric(freq_target, freq_weight, fmin, fmax, agg)
        mt = tuple(mt..., freq)
    end

    tspec = isempty(tuning_params) || prob === nothing ? 
        TuningSpec(Dict()) : 
        build_tuning_spec(prob, string(name), tuning_params)

    local N = length(mt)
    return Population{typeof(blox), N, typeof(mt)}(name, blox, mt, tspec, tunable)
end

"""
    update_parameters!(prob, populations, p, get_ps, set_ps!)

Update the `prob` in place, assigning the parameter values from `p` 
according to each population's `tuning` spec.
"""
function update_parameters!(prob, populations, p, get_ps, set_ps!)
    ps_new = get_ps(prob)
    offset = 1

    for pop in populations
        if !pop.tunable
            continue
        end

        for (param_name, inds) in pop.tuning.param_map
            ps_new[inds] .= abs.(p[offset]) 
            offset += 1
        end
    end

    set_ps!(prob, ps_new)
    return nothing
end

#############################
# OptimizationConfig + Loss #
#############################

struct OptimizationConfig{P,PopT,GetterT,SetterT,SolverT,EnsembleAlgT,dtT,DiffeqkargsT}
    prob::P
    populations::PopT
    get_ps::GetterT
    set_ps!::SetterT
    solver::SolverT
    ensemblealg::EnsembleAlgT
    dt::dtT
    other_diffeq_kwargs::DiffeqkargsT
    trajectories::Int
    seed::Int
end

@unroll function sum_metrics_unrolled(pops, sol; logging=false)
    total_err = zero(eltype(sol))
    for pop in pops
        total_err += compute_metrics(pop, sol; logging=logging)
    end
    return total_err
end

"""
    loss(p, config::OptimizationConfig; logging=false)

Update parameters, solve the ensemble problem, 
compute total error, and optionally log each metric's value.
"""
function loss(p, config::OptimizationConfig; logging::Bool=false)
    # Set random seed
    Random.seed!(config.seed)

    # Update prob in-place
    update_parameters!(config.prob, config.populations, p, config.get_ps, config.set_ps!)

    # Solve
    ens_prob = EnsembleProblem(config.prob)
    sol = solve(ens_prob, config.solver, config.ensemblealg;
                trajectories=config.trajectories,
                dt=config.dt,
                saveat=config.dt,
                config.other_diffeq_kwargs...)

    # Sum errors from each population
    total_err = sum_metrics_unrolled(config.populations, sol; logging=logging)
    return total_err
end

#################
# Example usage #
#################

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

    @named msn = Striatum_MSN_Adam(
        namespace=global_ns, 
        N_inhib=N_MSN, 
        I_bg=1.153064742988923*ones(N_MSN), 
        σ=0.17256774881503584
    )
    @named fsi = Striatum_FSI_Adam(
        namespace=global_ns, 
        N_inhib=N_FSI, 
        I_bg=6.196201739395473*ones(N_FSI), 
        σ=0.9548801242101033
    )
    @named gpe = GPe_Adam(
        namespace=global_ns, 
        N_inhib=N_GPe, 
        I_bg=3.272893843123162*ones(N_GPe), 
        σ=1.0959782801317943
    )
    @named stn = STN_Adam(
        namespace=global_ns, 
        N_exci=N_STN, 
        I_bg=2.2010777359961953*ones(N_STN), 
        σ=2.9158528502583545
    )

    g = MetaDiGraph()
    add_edge!(g, fsi => msn, weight=weight_FSI_MSN, connection_matrix=conn_FSI_MSN)
    add_edge!(g, msn => gpe, weight=weight_MSN_GPe, connection_matrix=conn_MSN_GPe)
    add_edge!(g, gpe => stn, weight=weight_GPe_STN, connection_matrix=conn_GPe_STN)
    add_edge!(g, stn => fsi, weight=weight_STN_FSI, connection_matrix=conn_STN_FSI)

    @info "Creating system from graph"
    @named sys = system_from_graph(g)

    tspan = (0.0, T)
    @info "Creating SDEProblem"
    prob = SDEProblem{true}(sys, [], tspan, [])

    return prob, sys, msn, fsi, gpe, stn
end

# Example
prob, sys, msn, fsi, gpe, stn = create_problem(0.1, 5500.0)

msn_pop = Population(
    :msn, msn; 
    frm   = (1.21, 60.0, -35.0, 200.0),   # FiringRateMetric: (target, weight, threshold, transient)
    freqm = (10.0, 0.5, 3.0, 20.0, "I_syn_msn"),   # FrequencyMetric: (target, weight, fmin, fmax, aggregate_state)
    prob = prob,   
    tunable = false
)

fsi_pop = Population(
    :fsi, fsi;
    frm   = (13.0, 10.0, -35.0, 200.0),
    freqm = (61.14, 1.0, 40.0, 90.0, "I_syn_fsi"), 
    prob = prob,
    tunable = false
)

gpe_pop = Population(
    :gpe, gpe;
    freqm = (85.0, 0.5, 40.0, 90.0, "V"), 
    tuning_params = ["I_bg", "σ"],
    prob = prob,
    tunable = true
)

stn_pop = Population(
    :stn, stn;
    tuning_params = ["I_bg", "σ"],
    prob = prob,
    tunable = false
)

other_diffeq_kwargs = (abstol=1e-3, reltol=1e-6, maxiters=1e10)
get_ps = getp(prob, tunable_parameters(sys)) 
set_ps! = setp(prob, tunable_parameters(sys))

config = OptimizationConfig(
    prob,
    (msn_pop, fsi_pop, gpe_pop, stn_pop),
    get_ps, 
    set_ps!,
    RKMil(),
    EnsembleThreads(),
    0.1,            
    other_diffeq_kwargs,
    3,               
    1234             
)

u = [3.272893843123162, 1.0959782801317943, 2.2010777359961953, 2.9158528502583545]
# optprob = Optimization.OptimizationProblem(loss, p0, config)
optprob = Optimization.OptimizationProblem((p, config)->loss(p, config; logging=true), p0, config)
callback = function (state, l)
    println("\n")
    @info "Iteration: $(state.iter)"
    @info "Parameters: $(state.u)"
    @info "Loss: $l"
    println("\n")
    return false
end

# Example run
res = solve(optprob, Optim.NelderMead(); 
    maxiters=2, 
    callback=callback
)