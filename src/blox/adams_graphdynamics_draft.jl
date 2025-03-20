using Neuroblox
using GraphDynamics

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

GraphDynamicsInterop.initialize_input(s::Subsystem{AdamNMDAR}) = (; Glu = 0.0, V = 0.0)

function GraphDynamicsInterop.subsystem_differential(s::Subsystem{AdamNMDAR}, inputs, t)
    # Unpack
    (; Glu, V) = inputs
    (; C, C_A, C_AA, D_AA, O_AA, O_AAB, C_AAB, D_AAB, C_AB, C_B) = s
    (; k_on, k_off, k_r, k_d, k_unblock, k_block, α, β) = s
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