using Neuroblox
using OrdinaryDiffEq
using Random, Distributions

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

NE = 80
NI = 20

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(1.5, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.1, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        nmda = AdamNMDAR(name=Symbol("NMDA$(ne.name)_$(ni.name)"), k_unblock=5.4)
        add_edge!(g, ne => nmda; weight=1.0)
        add_edge!(g, ni => nmda; weight=1.0, reverse=true)
        add_edge!(g, nmda => ne; weight=8.5)

        ampa = AdamAMPA(name=Symbol("AMPA$(ne.name)_$(ni.name)"))
        add_edge!(g, ne => ampa; weight=1)
        add_edge!(g, ampa => ni; weight=ḡₑ/NE)
    end
end

for ne_src ∈ exci
    for ne_dst ∈ exci
        nmda = AdamNMDAR(name=Symbol("NMDA$(ne_src.name)_$(ne_dst.name)"), k_unblock=5.4)
        add_edge!(g, ne_src => nmda; weight=1.0)
        add_edge!(g, ne_dst => nmda; weight=1.0, reverse=true)
        add_edge!(g, nmda => ne_dst; weight=8.5)

        ampa = AdamAMPA(name=Symbol("AMPA$(ne_src.name)_$(ne_dst.name)"))
        add_edge!(g, ne_src => ampa; weight=1)
        add_edge!(g, ampa => ne_dst; weight=ḡₑ/NE)
    end
end

for ni ∈ inhi
    for ne ∈ exci[1:20]
        GABA = AdamGABA(name=Symbol("GABA$(ni.name)_$(ne.name)"))
        add_edge!(g, ni => GABA; weight=1)
        add_edge!(g, GABA => ne; weight=ḡᵢ/NI)
    end
end

tspan = (0.0, 1000.0)
sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=0.5)
