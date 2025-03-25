using Neuroblox
using GraphDynamics

using Neuroblox.GraphDynamicsInterop

GraphDynamicsInterop.issupported(::AdamPYR) = true
GraphDynamicsInterop.components(v::AdamPYR) = (v,)

function GraphDynamicsInterop.to_subsystem(v::AdamPYR)
    C = GraphDynamicsInterop.recursive_getdefault(v.C)
    Eₙₐ = GraphDynamicsInterop.recursive_getdefault(v.Eₙₐ)
    ḡₙₐ = GraphDynamicsInterop.recursive_getdefault(v.ḡₙₐ)
    Eₖ = GraphDynamicsInterop.recursive_getdefault(v.Eₖ)
    ḡₖ = GraphDynamicsInterop.recursive_getdefault(v.ḡₖ)
    Eₗ = GraphDynamicsInterop.recursive_getdefault(v.Eₗ)
    ḡₗ = GraphDynamicsInterop.recursive_getdefault(v.ḡₗ)
    Iₐₚₚ = GraphDynamicsInterop.recursive_getdefault(v.Iₐₚₚ)
    Iₙₒᵢₛₑ = GraphDynamicsInterop.recursive_getdefault(v.Iₙₒᵢₛₑ)

    params = SubsystemParams{AdamPYR}(; C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ)

    V = GraphDynamicsInterop.recursive_getdefault(v.V)
    m = GraphDynamicsInterop.recursive_getdefault(v.m)
    h = GraphDynamicsInterop.recursive_getdefault(v.h)
    n = GraphDynamicsInterop.recursive_getdefault(v.n)

    states = SubsystemStates{AdamPYR}(; V, m, h, n) 

    Subsystem(states, params)
end

GraphDynamicsInterop.initialize_input(s::Subsystem{AdamPYR}) = (; jcn = 0.0)

function GraphDynamicsInterop.subsystem_differential(s::Subsystem{AdamPYR}, inputs, t)
    (; jcn) = inputs
    (; C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ) = s
    (; V, m, h, n) = s 

    αₘ(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
    βₘ(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
    αₕ(v) = 0.128*exp((v+50.0)/18.0)
    βₕ(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
    αₙ(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
    βₙ(v) = 0.5*exp(-(v+57.0)/40.0)

    m∞(v) = αₘ(v)/(αₘ(v) + βₘ(v))
    h∞(v) = αₕ(v)/(αₕ(v) + βₕ(v))
    n∞(v) = αₙ(v)/(αₙ(v) + βₙ(v))

    τₘ(v) = 1.0/(αₘ(v) + βₘ(v))
    τₕ(v) = 1.0/(αₕ(v) + βₕ(v))
    τₙ(v) = 1.0/(αₙ(v) + βₙ(v))

    return SubsystemStates{AdamPYR}(
        #=d/dt=# V = (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
        #=d/dt=# m = (m∞(V) - m)/τₘ(V),
        #=d/dt=# h = (h∞(V) - h)/τₕ(V),
        #=d/dt=# n = (n∞(V) - n)/τₙ(V)
    )
end

GraphDynamicsInterop.issupported(::AdamINP) = true
GraphDynamicsInterop.components(v::AdamINP) = (v,)

function GraphDynamicsInterop.to_subsystem(v::AdamINP)
    C = GraphDynamicsInterop.recursive_getdefault(v.C)
    Eₙₐ = GraphDynamicsInterop.recursive_getdefault(v.Eₙₐ)
    ḡₙₐ = GraphDynamicsInterop.recursive_getdefault(v.ḡₙₐ)
    Eₖ = GraphDynamicsInterop.recursive_getdefault(v.Eₖ)
    ḡₖ = GraphDynamicsInterop.recursive_getdefault(v.ḡₖ)
    Eₗ = GraphDynamicsInterop.recursive_getdefault(v.Eₗ)
    ḡₗ = GraphDynamicsInterop.recursive_getdefault(v.ḡₗ)
    Iₐₚₚ = GraphDynamicsInterop.recursive_getdefault(v.Iₐₚₚ)
    Iₙₒᵢₛₑ = GraphDynamicsInterop.recursive_getdefault(v.Iₙₒᵢₛₑ)

    params = SubsystemParams{AdamINP}(; C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ)

    V = GraphDynamicsInterop.recursive_getdefault(v.V)
    m = GraphDynamicsInterop.recursive_getdefault(v.m)
    h = GraphDynamicsInterop.recursive_getdefault(v.h)
    n = GraphDynamicsInterop.recursive_getdefault(v.n)

    states = SubsystemStates{AdamINP}(; V, m, h, n) 

    Subsystem(states, params)
end

GraphDynamicsInterop.initialize_input(s::Subsystem{AdamINP}) = (; jcn = 0.0)

function GraphDynamicsInterop.subsystem_differential(s::Subsystem{AdamINP}, inputs, t)
    (; jcn) = inputs
    (; C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ) = s
    (; V, m, h, n) = s 

    αₘ(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
    βₘ(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
    αₕ(v) = 0.128*exp((v+50.0)/18.0)
    βₕ(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
    αₙ(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
    βₙ(v) = 0.5*exp(-(v+57.0)/40.0)

    m∞(v) = αₘ(v)/(αₘ(v) + βₘ(v))
    h∞(v) = αₕ(v)/(αₕ(v) + βₕ(v))
    n∞(v) = αₙ(v)/(αₙ(v) + βₙ(v))

    τₘ(v) = 1.0/(αₘ(v) + βₘ(v))
    τₕ(v) = 1.0/(αₕ(v) + βₕ(v))
    τₙ(v) = 1.0/(αₙ(v) + βₙ(v))

    return SubsystemStates{AdamINP}(
        #=d/dt=# V = (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
        #=d/dt=# m = (m∞(V) - m)/τₘ(V),
        #=d/dt=# h = (h∞(V) - h)/τₕ(V),
        #=d/dt=# n = (n∞(V) - n)/τₙ(V)
    )
end

GraphDynamicsInterop.issupported(::AdamGABBA) = true
GraphDynamicsInterop.components(v::AdamGABBA) = (v,)

function GraphDynamicsInterop.to_subsystem(v::AdamGABBA)
    τᵢ = GraphDynamicsInterop.recursive_getdefault(v.τᵢ)

    params = SubsystemParams{AdamINP}(; τᵢ)

    sᵧ = GraphDynamicsInterop.recursive_getdefault(v.sᵧ)

    states = SubsystemStates{AdamINP}(; sᵧ) 

    Subsystem(states, params)
end

GraphDynamicsInterop.initialize_input(s::Subsystem{AdamGABBA}) = (; V = 0.0)

function GraphDynamicsInterop.subsystem_differential(s::Subsystem{AdamGABBA}, inputs, t)
    (; V) = inputs
    (; τᵢ) = s
    (; sᵧ) = s 

    gᵧ(v) = 2*(1+tanh(v/4))

    return SubsystemStates{AdamGABBA}(
        #=d/dt=# sᵧ = gᵧ(V)*(1-sᵧ) - sᵧ/τᵢ
    )
end

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
    Glu_max = GraphDynamicsInterop.recursive_getdefault(v.Glu_max)
    τ_Glu = GraphDynamicsInterop.recursive_getdefault(v.τ_Glu)
    θ = GraphDynamicsInterop.recursive_getdefault(v.θ)

    params = SubsystemParams{AdamNMDAR}(; k_on, k_off, k_r, k_d, k_unblock, k_block, α, β, Glu_max, τ_Glu, θ)

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
    Glu = GraphDynamicsInterop.recursive_getdefault(v.Glu)

    states = SubsystemStates{AdamNMDAR}(; C, C_A, C_AA, D_AA, O_AA, O_AAB, C_AAB, D_AAB, C_AB, C_B, Glu)

    Subsystem(states, params)
end

GraphDynamicsInterop.initialize_input(s::Subsystem{AdamNMDAR}) = (; jcn = 0.0, V = 0.0)

function heaviside(x)
    IfElse.ifelse(x > 0, 1.0, 0.0)
end

function GraphDynamicsInterop.subsystem_differential(s::Subsystem{AdamNMDAR}, inputs, t)
    # Unpack
    (; jcn, V) = inputs
    (; C, C_A, C_AA, D_AA, O_AA, O_AAB, C_AAB, D_AAB, C_AB, C_B, Glu) = s
    (; k_on, k_off, k_r, k_d, k_unblock, k_block, α, β, Glu_max, τ_Glu, θ) = s
    return SubsystemStates{AdamNMDAR}(
        #=d/dt=# C = k_off*C_A - 2*k_on*Glu*C,
        #=d/dt=# C_A = 2*k_off*C_AA + 2*k_on*Glu*C - (k_on*Glu + k_off)*C_A,
        #=d/dt=# C_AA = k_on*Glu*C_A + α*O_AA + k_r*D_AA - (2*k_off + β + k_d)*C_AA,
        #=d/dt=# D_AA = k_d*C_AA - k_r*D_AA,
        #=d/dt=# O_AA = β*C_AA + k_unblock*exp(V/47)*O_AAB - (α + k_block*exp(-V/17))*O_AA,
        #=d/dt=# O_AAB = k_block*exp(-V/17)*O_AA + β*C_AAB - (k_unblock*exp(V/47) + α)*O_AAB,
        #=d/dt=# C_AAB = α*O_AAB + k_on*Glu*C_AB + k_r*D_AAB - (β + 2*k_off + k_d)*C_AAB,
        #=d/dt=# D_AAB = k_d*C_AAB - k_r*D_AAB,
        #=d/dt=# C_AB = 2*k_off*C_AAB + 2*k_on*Glu*C_B - (k_on*Glu + k_off)*C_AB,
        #=d/dt=# C_B = k_off*C_AB - 2*k_on*Glu*C_B,
        #=d/dt=# Glu = Glu_max*heaviside(jcn - θ) - Glu/τ_Glu
    )
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

## Not GraphDynamics but work on connections
# Helper function for connections
function make_nmda_edge!(g, prenrn, postnrn)
    glu = AdamGlu(name=Symbol("Glu$(prenrn.name)_$(postnrn.name)"))
    nmda = AdamNMDAR(name=Symbol("NMDA$(prenrn.name)_$(postnrn.name)"))
    add_edge!(g, prenrn => glu; weight=1.0)
    add_edge!(g, glu => nmda; weight=1.0)
    add_edge!(g, postnrn => nmda; weight=1.0)
    add_edge!(g, nmda => postnrn; weight=8.5)
end