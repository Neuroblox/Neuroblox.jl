using Neuroblox
using Neuroblox: namespaced_nameof
using GraphDynamics
using Neuroblox.GraphDynamicsInterop: BasicConnection
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using Optimization
using OptimizationBBO
using CairoMakie

rule_type(::ConnectionMatrix{N, CR}) where {N, CR} = CR
function set_weight!(prob, name_src, name_dst, conn::Conn) where {Conn <: ConnectionRule}
    (; connection_matrices) = prob.p
    names_partitioned = prob.f.sys.names_partitioned
    i_src = i_dst = nothing
    j_src = j_dst = nothing
    for i ∈ eachindex(names_partitioned)
        for j ∈ eachindex(names_partitioned[i])
            if names_partitioned[i][j] == name_src
                i_src = i
                j_src = j
            end
            if names_partitioned[i][j] == name_dst
                i_dst = i
                j_dst = j
            end
        end
    end
    if any(isnothing, (i_src, j_src))
        error("No subsystem named $name_src was found in the problem.")
    end
    if any(isnothing, ( i_dst, j_dst))
        error("No subsystem named $name_dst was found in the problem.")
    end
    conn_types = [rule_type(connection_matrices[i]) for i ∈ 1:length(connection_matrices)]
    nc = findfirst(RT -> Conn <: RT, conn_types)
    if isnothing(nc)
        error("Problem has no connection matrix of eltype $Conn, available eltypes are $(join(conn_types, ", "))")
    end
    connection_matrices[nc][i_src, i_dst][j_src, j_dst] = conn
    prob
end

function bimodal_coeff(samples)
    N = length(samples)
    sk = skewness(samples)
    kr = kurtosis(samples)

    return (sk^2 + 1) / (kr + (3*(N - 1)^2) / ((N - 2) * (N - 3)))
end

function generate_target_bimodal_coeff()
    @named hh1 = HHNeuronExciBlox(; I_bg=1.8, G_syn=3)
    tspan = (0.0, 2000.0)
    IBI = 250 # inter-burst interval [ms]
    t_burst = 500 # burst duration
    cbs_stop = [
        [t] => [hh1.I_bg ~ 0]
        for t in t_burst:(t_burst + IBI):last(tspan)
    ]
    cbs_start = [
        [t] => [hh1.I_bg ~ 2]
        for t in (t_burst + IBI):(t_burst + IBI):last(tspan)
    ]

    sys = system(hh1; discrete_events=vcat(cbs_start, cbs_stop))
    prob = ODEProblem(sys, [], tspan)
    sol = solve(prob; saveat=0.05)
    st = inter_spike_intervals(hh1, sol; threshold=0)

    return bimodal_coeff(st)
end

function objective(x, sys, PYR_neurons, currents, weights, tspan, target_bimodal_coeff)
    I_bg_PYR, I_bg_INP, I_bg_INT, w_PYR_INP, w_INP_INP, w_INT_INP, w_PYR_PYR, w_INP_PYR, w_INT_PYR = x

    p = vcat(
        [I_bg => I_bg_PYR for I_bg in currents[:I_bg_PYR]],
        [I_bg => I_bg_INP for I_bg in currents[:I_bg_INP]],
        [I_bg => I_bg_INT for I_bg in currents[:I_bg_INT]]
    )

    prob = ODEProblem(sys, [], tspan, p)

    for t in weights[:w_PYR_INP]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_PYR_INP))
    end

    for t in weights[:w_INP_INP]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_INP_INP))
    end

    for t in weights[:w_INT_INP]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_INT_INP))
    end

    for t in weights[:w_PYR_PYR]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_PYR_PYR))
    end

    for t in weights[:w_INP_PYR]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_INP_PYR))
    end

    for t in weights[:w_INT_PYR]
        name_src, name_dst = t
        set_weight!(prob, name_src, name_dst, BasicConnection(w_INT_PYR))
    end

    sol = solve(prob; saveat=0.05)

    st = flat_inter_spike_intervals(PYR_neurons, sol; threshold=0)
    bm = bimodal_coeff(st)

    return (bm - target_bimodal_coeff)^2
end


N_PYR = 80
N_PYR_PYR = 10
N_PYR_INP = 10

N_INP = 20
N_INP_PYR = 10
N_INP_INP = 10

N_INT = 80
N_INT_PYR = 5
N_INT_INP = 5

PYR = [HHNeuronExciBlox(name=Symbol("PYR$i"), namespace=:g, I_bg=1.8) for i in 1:N_PYR]
INP = [HHNeuronInhibBlox(name=Symbol("INP$i"), namespace=:g, I_bg=0.2) for i in 1:N_INP] 
INT = [HHNeuronInhibBlox(name=Symbol("INT$i"), namespace=:g, I_bg=-0.5) for i in 1:N_INT]

currents = Dict(
    :I_bg_PYR => Symbol.(["PYR$(i)₊I_bg" for i in 1:N_PYR]),
    :I_bg_INP => Symbol.(["INP$(i)₊I_bg" for i in 1:N_INP]),
    :I_bg_INT => Symbol.(["INT$(i)₊I_bg" for i in 1:N_INT])
)

weights = Dict(
    :w_PYR_INP => Tuple{Symbol, Symbol}[], 
    :w_INP_INP => Tuple{Symbol, Symbol}[], 
    :w_INT_INP => Tuple{Symbol, Symbol}[],
    :w_PYR_PYR => Tuple{Symbol, Symbol}[],
    :w_INP_PYR => Tuple{Symbol, Symbol}[],
    :w_INT_PYR => Tuple{Symbol, Symbol}[]
)

g = MetaDiGraph()

for ni_dst ∈ INP
    idx_PYR_src = sample(Base.OneTo(N_PYR), N_PYR_INP; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = MoradiNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ni_dst.name)"))
        add_edge!(g, ne_src => nmda; weight=1)
        add_edge!(g, ni_dst => nmda; weight=1, reverse=true)
        add_edge!(g, nmda => ni_dst; weight=1/N_PYR_INP)

        add_edge!(g, ne_src => ni_dst; weight=1/N_PYR_INP)

        push!(weights[:w_PYR_INP], (namespaced_nameof(ne_src), namespaced_nameof(ni_dst)))
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_INP; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INP_INP)

        push!(weights[:w_INP_INP], (namespaced_nameof(ni_src), namespaced_nameof(ni_dst)))
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_INP; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INT_INP)

        push!(weights[:w_INT_INP], (namespaced_nameof(ni_src), namespaced_nameof(ni_dst)))
    end
end

for (i, ne_dst) ∈ enumerate(PYR)
    idxs = [j for j in Base.OneTo(N_PYR) if j != i]
    idx_PYR_src = sample(idxs, N_PYR_PYR; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = MoradiNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ne_dst.name)"))
        add_edge!(g, ne_src => nmda; weight=1)
        add_edge!(g, ne_dst => nmda; weight=1, reverse=true)
        add_edge!(g, nmda => ne_dst; weight=1/N_PYR_PYR)

        add_edge!(g, ne_src => ne_dst; weight=1/N_PYR_PYR)

        push!(weights[:w_PYR_PYR], (namespaced_nameof(ne_src), namespaced_nameof(ne_dst)))
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_PYR; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INP_PYR)

        push!(weights[:w_INP_PYR], (namespaced_nameof(ni_src), namespaced_nameof(ne_dst)))
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_PYR; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INT_PYR)

        push!(weights[:w_INT_PYR], (namespaced_nameof(ni_src), namespaced_nameof(ne_dst)))
    end
end

tspan = (0.0, 2000.0)
sys = system_from_graph(g; name=:g, graphdynamics=true)

target_bimodal_coeff = generate_target_bimodal_coeff()

obj = OptimizationFunction(
    (p, hyperp) -> objective(p, sys, PYR, currents, weights, tspan, target_bimodal_coeff), 
    Optimization.AutoFiniteDiff()
)
p0 = [1.8, 0.2, -0.5, 1, 1, 1, 1, 1, 1]
prob = OptimizationProblem(obj, p0, lb=[-3, -3, -3, 0, 0, 0, 0, 0, 0], ub=[3, 3, 3, 20, 20, 20, 20, 20, 20])
sol = solve(prob, Optimization.LBFGS())

