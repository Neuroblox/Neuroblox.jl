using Neuroblox
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using CairoMakie

function connect_through_receptor!(g, source, destination, receptor; weight=1)
    add_edge!(g, source => receptor; weight=1)
    add_edge!(g, receptor => destination; weight)
end

function connect_through_receptor!(g, source, destination, receptor::AdamNMDAR; weight=1)
    add_edge!(g, source => receptor; weight=1)
    add_edge!(g, destination => receptor; weight=1, reverse=true)
    add_edge!(g, receptor => destination; weight)
end

ḡᵢ = 0.5
ḡₑ = 0.2

N_PYR = 80
N_PYR_PYR = 10
N_PYR_INP = 10

N_INP = 20
N_INP_PYR = 10
N_INP_INP = 10

N_INT = 80
N_INT_PYR = 5
N_INT_INP = 5

PYR = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(1.5, 0.05))) for i in 1:N_PYR]
INP = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.1, 0.05))) for i in 1:N_INP] 
INT = [AdamINP(name=Symbol("INT$i"), Iₐₚₚ=rand(Normal(0.1, 0.05))) for i in 1:N_INT]

g = MetaDiGraph()

for ni_dst ∈ INP
    idx_PYR_src = sample(Base.OneTo(N_PYR), N_PYR_INP; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = AdamNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ni_dst.name)"), k_unblock=10.4)
        add_edge!(g, ne_src => nmda; weight=1.0)
        add_edge!(g, ni_dst => nmda; weight=1.0, reverse=true)
        add_edge!(g, nmda => ne_src; weight=8.5)

        ampa = AdamAMPA(name=Symbol("AMPA_$(ne_src.name)_$(ni_dst.name)"))
        add_edge!(g, ne_src => ampa; weight=1)
        add_edge!(g, ampa => ni_dst; weight=ḡₑ/N_PYR_INP)
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_INP; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        GABA = AdamGABA(name=Symbol("GABA_$(ni_src.name)_$(ni_dst.name)"))
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ni_dst; weight=ḡᵢ/N_INP_INP)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_INP; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        GABA = AdamGABA(name=Symbol("GABA_$(ni_src.name)_$(ni_dst.name)"))
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ni_dst; weight=ḡᵢ/N_INT_INP)
    end
end

for (i, ne_dst) ∈ enumerate(PYR)
    idxs = [j for j in Base.OneTo(N_PYR) if j != i]
    idx_PYR_src = sample(idxs, N_PYR_PYR; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = AdamNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ne_dst.name)"), k_unblock=10.4)
        add_edge!(g, ne_src => nmda; weight=1.0)
        add_edge!(g, ne_dst => nmda; weight=1.0, reverse=true)
        add_edge!(g, nmda => ne_dst; weight=8.5)

        ampa = AdamAMPA(name=Symbol("AMPA_$(ne_src.name)_$(ne_dst.name)"))
        add_edge!(g, ne_src => ampa; weight=1)
        add_edge!(g, ampa => ne_dst; weight=ḡₑ/N_PYR_PYR)
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_PYR; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        GABA = AdamGABA(name=Symbol("GABA_$(ni_src.name)_$(ne_dst.name)"))
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ne_dst; weight=ḡᵢ/N_INP_PYR)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_PYR; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        GABA = AdamGABA(name=Symbol("GABA_$(ni_src.name)_$(ne_dst.name)"))
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ne_dst; weight=ḡᵢ/N_INT_PYR)
    end
end

tspan = (0.0, 1000.0)
sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5())

rasterplot(PYR, sol; threshold=-10)
