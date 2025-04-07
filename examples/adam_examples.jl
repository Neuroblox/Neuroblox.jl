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

N_PYR = 80
N_PYR_PYR = 10
N_PYR_INP = 10

N_INP = 20
N_INP_PYR = 10
N_INP_INP = 10

N_INT = 80
N_INT_PYR = 5
N_INT_INP = 5

PYR = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=-0.25) for i in 1:N_PYR]
INP = [AdamIN(name=Symbol("INP$i"), Iₐₚₚ=0.1) for i in 1:N_INP] 
INT = [AdamIN(name=Symbol("INT$i"), Iₐₚₚ=-1.4) for i in 1:N_INT]

g = MetaDiGraph()

for ni_dst ∈ INP
    idx_PYR_src = sample(Base.OneTo(N_PYR), N_PYR_INP; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = AdamNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ni_dst.name)"), k_unblock=3.8, g=9.5)
        add_edge!(g, ne_src => nmda; weight=1)
        add_edge!(g, ni_dst => nmda; weight=1, reverse=true)
        add_edge!(g, nmda => ni_dst; weight=1/N_PYR_PYR)

        ampa = AdamAMPA(name=Symbol("AMPA_$(ne_src.name)_$(ni_dst.name)"))
        add_edge!(g, ne_src => ampa; weight=1)
        add_edge!(g, ampa => ni_dst; weight=1/N_PYR_INP)
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_INP; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        GABA = AdamGABA(
            name=Symbol("GABA_$(ni_src.name)_$(ni_dst.name)"),
            g = 0.8
        )
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ni_dst; weight=1/N_INP_INP)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_INP; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        GABA = AdamGABA(
            name=Symbol("GABA_$(ni_src.name)_$(ni_dst.name)"),
            g = 5
        )
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ni_dst; weight=1/N_INT_INP)
    end
end

for (i, ne_dst) ∈ enumerate(PYR)
    idxs = [j for j in Base.OneTo(N_PYR) if j != i]
    idx_PYR_src = sample(idxs, N_PYR_PYR; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = AdamNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ne_dst.name)"), k_unblock=3.8, g=8.5)
        add_edge!(g, ne_src => nmda; weight=1)
        add_edge!(g, ne_dst => nmda; weight=1, reverse=true)
        add_edge!(g, nmda => ne_dst; weight=1/N_PYR_PYR)

        ampa = AdamAMPA(name=Symbol("AMPA_$(ne_src.name)_$(ne_dst.name)"))
        add_edge!(g, ne_src => ampa; weight=1)
        add_edge!(g, ampa => ne_dst; weight=1/N_PYR_PYR)
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_PYR; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        GABA = AdamGABA(
            name=Symbol("GABA_$(ni_src.name)_$(ne_dst.name)"),
            g = 0.8
        )
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ne_dst; weight=1/N_INP_PYR)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_PYR; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        GABA = AdamGABA(
            name=Symbol("GABA_$(ni_src.name)_$(ne_dst.name)"),
            g = 5
        )
        add_edge!(g, ni_src => GABA; weight=1)
        add_edge!(g, GABA => ne_dst; weight=1/N_INT_PYR)
    end
end

tspan = (0.0, 5000.0)
sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(); saveat=0.05)

rasterplot(PYR, sol; threshold=-10)
