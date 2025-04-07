using Neuroblox
using GraphDynamics

using Neuroblox.GraphDynamicsInterop
using Neuroblox.GraphDynamicsInterop: BasicConnection

function (c::BasicConnection)(sys_src::Subsystem{AdamPYR}, sys_dst::Subsystem{<:AdamAMPA})
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.V = sys_src.V
    
    return acc
end

function (c::BasicConnection)(sys_src::Subsystem{AdamAMPA}, sys_dst::Subsystem{<:AbstractAdamNeuron})
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.jcn = c.weight * sys_src.sₐₘₚₐ * sys_src.g * (sys_dst.V - sys_src.E)
    
    return acc
end

function (c::BasicConnection)(sys_src::Subsystem{AdamIN}, sys_dst::Subsystem{<:AdamGABA})
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.V = sys_src.V
    
    return acc
end

function (c::BasicConnection)(sys_src::Subsystem{AdamGABA}, sys_dst::Subsystem{<:AbstractAdamNeuron})
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.jcn = c.weight * sys_src.sᵧ * sys_src.g * (sys_dst.V - sys_src.E)
    
    return acc
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{<:Neuroblox.AbstractAdamNeuron}, sys_dst::Subsystem{AdamNMDAR}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.jcn = sys_src.V
    
    return acc
end

function (c::GraphDynamicsInterop.ReverseConnection)(sys_src::Subsystem{<:Neuroblox.AbstractAdamNeuron}, sys_dst::Subsystem{AdamNMDAR}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.V = sys_src.V
    
    return acc
end

function (c::GraphDynamicsInterop.BasicConnection)(sys_src::Subsystem{AdamNMDAR}, sys_dst::Subsystem{<:Neuroblox.AbstractAdamNeuron}, t)
    acc = GraphDynamicsInterop.initialize_input(sys_dst)
    acc = @set acc.jcn = c.weight * sys_src.g * sys_src.O_AA * (sys_dst.V - sys_src.E)
    
    return acc
end

## Not GraphDynamics but work on connections
# Helper function for connections
function make_nmda_edge!(g, prenrn, postnrn)
    nmda = AdamNMDAR(name=Symbol("NMDA$(prenrn.name)_$(postnrn.name)"))
    add_edge!(g, prenrn => nmda; weight=1.0)
    add_edge!(g, postnrn => nmda; weight=1.0)
    add_edge!(g, nmda => postnrn; weight=8.5)
end