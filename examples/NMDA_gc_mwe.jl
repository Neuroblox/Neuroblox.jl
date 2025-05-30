using Neuroblox
using Random
using Distributions
using StatsBase

N_PYR = 80
N_PYR_PYR = 10
N_PYR_INP = 10

N_INP = 20
N_INP_PYR = 10
N_INP_INP = 10

N_INT = 80
N_INT_PYR = 5
N_INT_INP = 5

PYR = [HHNeuronExciBlox(name=Symbol("PYR$i"), I_bg=2) for i in 1:N_PYR]
INP = [HHNeuronInhibBlox(name=Symbol("INP$i"), I_bg=0.2) for i in 1:N_INP] 
INT = [HHNeuronInhibBlox(name=Symbol("INT$i"), I_bg=2) for i in 1:N_INT]

g = MetaDiGraph()

for ni_dst ∈ INP
    idx_PYR_src = sample(Base.OneTo(N_PYR), N_PYR_INP; replace = false)
    for ne_src ∈ PYR[idx_PYR_src]
        nmda = MoradiNMDAR(name=Symbol("NMDA_$(ne_src.name)_$(ni_dst.name)"))
        add_edge!(g, ne_src => nmda; weight=1)
        add_edge!(g, ni_dst => nmda; weight=1, reverse=true)
        add_edge!(g, nmda => ni_dst; weight=1/N_PYR_INP)

        add_edge!(g, ne_src => ni_dst; weight=1/N_PYR_INP)
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_INP; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INP_INP)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_INP; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ni_dst; weight=1/N_INT_INP)
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
    end

    idx_INP_src = sample(Base.OneTo(N_INP), N_INP_PYR; replace = false)
    for ni_src ∈ INP[idx_INP_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INP_PYR)
    end

    idx_INT_src = sample(Base.OneTo(N_INT), N_INT_PYR; replace = false)
    for ni_src ∈ INT[idx_INT_src]
        add_edge!(g, ni_src => ne_dst; weight=1/N_INT_PYR)
    end
end

tspan = (0.0, 50.0)
for _ ∈ 1:10
    sys = system_from_graph(g, graphdynamics=true)
end
