using Neuroblox
using Neuroblox: namespaced_nameof
using GraphDynamics
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using Optimization
using OptimizationBBO
using CairoMakie

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

function objective(x, prob, PYR_neurons, currents, weights, tspan, target_bimodal_coeff)
    I_bg_PYR, I_bg_INP, I_bg_INT, w_PYR_INP, w_INP_INP, w_INT_INP, w_PYR_PYR, w_INP_PYR, w_INT_PYR = x

    p = vcat(
        [I_bg => I_bg_PYR for I_bg in currents[:I_bg_PYR]],
        [I_bg => I_bg_INP for I_bg in currents[:I_bg_INP]],
        [I_bg => I_bg_INT for I_bg in currents[:I_bg_INT]],
        [w => w_PYR_INP for w in weights[:w_PYR_INP]],
        [w => w_INP_INP for w in weights[:w_INP_INP]],
        [w => w_INT_INP for w in weights[:w_INT_INP]],
        [w => w_PYR_PYR for w in weights[:w_PYR_PYR]],
        [w => w_INP_PYR for w in weights[:w_INP_PYR]],
        [w => w_INT_PYR for w in weights[:w_INT_PYR]],
    )

    prob_new = remake(prob; p = p)
    
    sol = solve(prob_new, Rodas4P(); saveat=0.05, abstol=1e-6, reltol=1e-6)

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
    :w_PYR_INP => Symbol[], 
    :w_INP_INP => Symbol[],
    :w_INT_INP => Symbol[],
    :w_PYR_PYR => Symbol[],
    :w_INP_PYR => Symbol[],
    :w_INT_PYR => Symbol[],
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

        push!(weights[:w_PYR_INP], Symbol(:w_, namespaced_nameof(ne_src), :_, namespaced_nameof(ni_dst)))
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_INP; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INP_INP)

        push!(weights[:w_INP_INP], Symbol(:w_, namespaced_nameof(ni_src), :_, namespaced_nameof(ni_dst)))
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_INP; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INT_INP)

        push!(weights[:w_INT_INP], Symbol(:w_, namespaced_nameof(ni_src), :_, namespaced_nameof(ni_dst)))
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

        push!(weights[:w_PYR_PYR], Symbol(:w_, namespaced_nameof(ne_src), :_, namespaced_nameof(ne_dst)))
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_PYR; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INP_PYR)

        push!(weights[:w_INP_PYR], Symbol(:w_, namespaced_nameof(ni_src), :_, namespaced_nameof(ne_dst)))
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_PYR; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INT_PYR)

        push!(weights[:w_INT_PYR], Symbol(:w_, namespaced_nameof(ni_src), :_, namespaced_nameof(ne_dst)))
    end
end

tspan = (0.0, 2000.0)
sys = system_from_graph(g; name=:g, graphdynamics=true)
target_bimodal_coeff = generate_target_bimodal_coeff()

# make the arguments to `objective` local variables before passing them to the closure. Otherwise, the closure will be slower
# due to referencing non-constant global variables
obj = let prob_inner = ODEProblem(sys, [], tspan, []),
    PYR = PYR,
    currents = currents,
    weights = weights,
    target_bimodal_coeff = target_bimodal_coeff
    
    OptimizationFunction(
        (p, hyperp) -> objective(p, prob_old, PYR, currents, weights, tspan, target_bimodal_coeff), 
        Optimization.AutoFiniteDiff()
    )
end
p0 = [1.8, 0.2, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
prob = OptimizationProblem(obj, p0, lb=[-3, -3, -3, 0, 0, 0, 0, 0, 0], ub=[3, 3, 3, 3, 3, 3, 3, 3, 3])

#alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
alg = Optimization.LBFGS()
sol = solve(prob, alg)

