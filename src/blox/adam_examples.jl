using Neuroblox
using OrdinaryDiffEq
using Random, Distributions
using Plots
using GraphDynamics
#import Neuroblox: AbstractNeuronBlox, paramscoping
#using BenchmarkTools

ḡᵢ = 0.5
ḡₑ = 0.2

NI = 20
NE = 80

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(0.25, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.3, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        add_edge!(g, ne => ni; weight=ḡₑ/NE)
        add_edge!(g, ni => ne; weight=ḡᵢ/NI)
    end
end

tspan = (0.0, 500.0)
# begin
#     @btime @named sys = system_from_graph(g, graphdynamics=true)
#     @btime prob = ODEProblem(sys, [], tspan)
#     @btime sol = solve(prob, Tsit5(), saveat=0.5)
# end

@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=0.5)

plot(sol, idxs=1:5:(NE+NI)*5)

using Neuroblox.GraphDynamicsInterop
GraphDynamicsInterop.issupported(::AdamNMDAR) = true
GraphDynamicsInterop.components(v::AdamNMDAR) = (v,)

function GraphDynamicsInterop.to_subsystem(v::AdamNMDAR)
    # Extract default parameter values
    k_on = GraphDynamicsInterop.recursive_getdefault(v.system.k_on)
    k_off = GraphDynamicsInterop.recursive_getdefault(v.system.k_off)
    k_r = GraphDynamicsInterop.recursive_getdefault(v.system.k_r)
    k_d = GraphDynamicsInterop.recursive_getdefault(v.system.k_d)
    k_unblock = GraphDynamicsInterop.recursive_getdefault(v.system.k_unblock)
    k_block = GraphDynamicsInterop.recursive_getdefault(v.system.k_block)
    α = GraphDynamicsInterop.recursive_getdefault(v.system.α)
    β = GraphDynamicsInterop.recursive_getdefault(v.system.β)

    params = SubsystemParams{AdamNMDAR}(; k_on, k_off, k_r, k_d, k_unblock, k_block, α, β)

    # Extract the default values of states
    C = GraphDynamicsInterop.recursive_getdefault(v.system.C)
    C_A = GraphDynamicsInterop.recursive_getdefault(v.system.C_A)
    C_AA = GraphDynamicsInterop.recursive_getdefault(v.system.C_AA)
    D_AA = GraphDynamicsInterop.recursive_getdefault(v.system.D_AA)
    O_AA = GraphDynamicsInterop.recursive_getdefault(v.system.O_AA)
    O_AAB = GraphDynamicsInterop.recursive_getdefault(v.system.O_AAB)
    C_AAB = GraphDynamicsInterop.recursive_getdefault(v.system.C_AAB)
    D_AAB = GraphDynamicsInterop.recursive_getdefault(v.system.D_AAB)
    C_AB = GraphDynamicsInterop.recursive_getdefault(v.system.C_AB)
    C_B = GraphDynamicsInterop.recursive_getdefault(v.system.C_B)
    states = SubsystemStates{AdamNMDAR}(; C, C_A, C_AA, D_AA, O_AA, O_AAB, C_AAB, D_AAB, C_AB, C_B)

    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{AdamNMDAR}) = (; Glu = 0.0, V = 0.0)

function GraphDynamics.subsystem_differential(s::Subsystem{AdamNMDAR}, inputs, t)
    # Unpack
    (; Glu, V) = inputs
    (; C, C_A, C_AA, D_AA, O_AA, O_AAB, C_AAB, D_AAB, C_AB, C_B) = s
    (; k_on, k_off, k_r, k_d, k_unblock, k_block, α, β) = s
    return SubsystemStates{AdamNMDAR}(
        #=d/dt=# C = k_off*C_A - 2*k_on*Glu*C,
        #=d/dt=# C_A = 2*k_off*C_AA + 2*k_on*Glu*C - (k_on*Glu + k_off)*C_A,
        #=d/dt=# C_AA = k_on*Glu*C_A + α*O_AA + k_r*D_AA - (2*k_off + β + k_d)*C_AA,
        #=d/dt=# D_AA = k_d*C_AA - k_r*D_AA,
        #=d/dt=# O_AA = k_r*D_AA - α*O_AA,
        #=d/dt=# O_AAB = k_unblock*C_AAB - k_block*O_AAB,
        #=d/dt=# C_AAB = k_block*O_AAB - k_unblock*C_AAB,
        #=d/dt=# D_AAB = k_d*C_AAB - k_r*D_AAB,
        #=d/dt=# C_AB = k_off*C_AAB - 2*k_on*Glu*C_AB,
        #=d/dt=# C_B = k_off*C_AB - 2*k_on*Glu*C_B
    )
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{AdamGlu}, sys_dst::Subsystem{AdamNMDAR}, t)
    w = c.weight
    Glu = sys_src.Glu
    V = 0.0
    (; Glu, V)
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{<:Neuroblox.AbstractAdamNeuron}, sys_dst::Subsystem{AdamNMDAR}, t)
    w = c.weight
    Glu = 0.0
    V = sys_src.V
    (; Glu, V)
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{AdamNMDAR}, sys_dst::Subsystem{<:Neuroblox.AbstractAdamNeuron}, t)
    w = c.weight
    O_AA = sys_src.O_AA
    jcn = w*O_AA*sys_dst.V
    (; jcn)
end

exci = AdamPYR(name=:PYR, Iₐₚₚ=0.25)
glur = AdamGlu(name=:Glu, θ=-59.0)
nmda = AdamNMDAR(name=:NMDA)
exci2 = AdamPYR(name=:PYR2, Iₐₚₚ=0.33)

g = MetaDiGraph()
add_edge!(g, exci => glur; weight=1.0)
add_edge!(g, glur => nmda; weight=1.0)
add_edge!(g, exci2 => nmda; weight=1.0)
add_edge!(g, nmda => exci2; weight=1.0)

tspan = (0.0, 500.0)
@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=0.5)
plot(sol)

NE = 800
NI = 800

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(0.25, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.3, 0.05))) for i in 1:NI] # bump up to 0.3
nmdar = [AdamNMDAR(name=Symbol("NMDA$i")) for i in 1:NE]
glu = [AdamGlu(name=Symbol("Glu$i")) for i in 1:NI]

g = MetaDiGraph()

for i in axes(exci, 1)
    add_edge!(g, exci[i] => glu[i]; weight=1.0)
    add_edge!(g, glu[i] => nmdar[i]; weight=1.0)
    add_edge!(g, inhi[i] => nmdar[i]; weight=1.0)
    add_edge!(g, nmdar[i] => inhi[i]; weight=1.0)
end

tspan = (0.0, 500.0)
@time @named sys = system_from_graph(g, graphdynamics=true)
@time prob = ODEProblem(sys, [], tspan)
@time sol = solve(prob, Tsit5(), saveat=0.5)

function make_nmda_edge!(g, prenrn, postnrn)
    glu = AdamGlu(name=Symbol("Glu$(prenrn.name)_$(postnrn.name)"))
    nmda = AdamNMDAR(name=Symbol("NMDA$(prenrn.name)_$(postnrn.name)"))
    add_edge!(g, prenrn => glu; weight=1.0)
    add_edge!(g, glu => nmda; weight=1.0)
    add_edge!(g, postnrn => nmda; weight=1.0)
    add_edge!(g, nmda => postnrn; weight=1.0)
end

NE = 80
NI = 20

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(1.5, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.1, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        make_nmda_edge!(g, ne, ni)
    end
end

for ne ∈ exci
    for ne ∈ exci
        add_edge!(g, ne => ne; weight=1.0)
    end
end

for ni ∈ inhi
    for ne ∈ exci[1:20]
        add_edge!(g, ni => ne; weight=ḡᵢ/NI)
    end
end

tspan = (0.0, 500.0)
@time @named sys = system_from_graph(g, graphdynamics=true)
@time prob = ODEProblem(sys, [], tspan)
@time sol = solve(prob, Tsit5(), saveat=0.5)

## Testing Glu for threshold setting
## Commented out for now but useful for tuning later so leaving in the file
# exci = AdamPYR(name=:PYR, Iₐₚₚ=0.25)
# glur = AdamGlu(name=:Glu, θ=-59.0)

# g = MetaDiGraph()
# add_edge!(g, exci => glur; weight=1.0)

# tspan = (0.0, 500.0)
# @named sys = system_from_graph(g, graphdynamics=false)
# prob = ODEProblem(sys, [], tspan)
# sol = solve(prob, Tsit5(), saveat=0.5)
# plot(sol, idxs=6)