const GD = GraphDynamics


@blox struct ConstantDAInput(; name, namespace=nothing, DA=0.0) <: AbstractStimulus
    @params DA
    @states
    @inputs
    @outputs DA
    @equations begin end
end

@blox struct ConstantIStimInput(; name, namespace=nothing, I=0.0) <: AbstractStimulus
    @params I
    @states
    @inputs
    @outputs I
    @equations begin end
end

@blox struct ConstantIAppInput(; name, namespace=nothing, I_app=0.0) <: AbstractStimulus
    @params I_app
    @states
    @inputs
    @outputs I_app
    @equations begin end
end

@blox struct ConstantAChInput(; name, namespace=nothing, ACh=0.0) <: AbstractStimulus
    @params ACh
    @states
    @inputs
    @outputs ACh
    @equations begin end
end

@blox struct ConstantICaInput(; name, namespace=nothing, I_Ca=0.0) <: AbstractStimulus
    @params I_Ca
    @states
    @inputs
    @outputs I_Ca
    @equations begin end
end

@blox struct ConstantCaBulkInput(; name, namespace=nothing, Ca_bulk=0.05) <: AbstractStimulus
    @params Ca_bulk
    @states
    @inputs
    @outputs Ca_bulk
    @equations begin end
end

@blox struct ConstantCChInput(; name, namespace=nothing, CCh=0.0) <: AbstractStimulus
    @params CCh
    @states
    @inputs
    @outputs CCh
    @equations begin end
end

@blox struct ConstantModeInput(; name, namespace=nothing, mode=0.0) <: AbstractStimulus
    @params mode
    @states
    @inputs
    @outputs mode
    @equations begin end
end

@blox struct ConstantMuscarinicInput(; name, namespace=nothing, M=0.0) <: AbstractStimulus
    @params M
    @states
    @inputs
    @outputs M
    @equations begin end
end

@blox struct ConstantVPreInput(; name, namespace=nothing, V_pre=0.0) <: AbstractStimulus
    @params V_pre
    @states
    @inputs
    @outputs V_pre
    @equations begin end
end

@blox struct ConstantVPostInput(; name, namespace=nothing, V_post=0.0) <: AbstractStimulus
    @params V_post
    @states
    @inputs
    @outputs V_post
    @equations begin end
end

@blox struct ConstantMNMDA1Input(; name, namespace=nothing, M_NMDA1=1.0) <: AbstractStimulus
    @params M_NMDA1
    @states
    @inputs
    @outputs M_NMDA1
    @equations begin end
end

@blox struct ConstantGAsympInput(; name, namespace=nothing, G_asymp=0.0) <: AbstractStimulus
    @params G_asymp
    @states
    @inputs
    @outputs G_asymp
    @equations begin end
end

@blox struct ConstantMAMPA2Input(; name, namespace=nothing, M_AMPA2=1.0) <: AbstractStimulus
    @params M_AMPA2
    @states
    @inputs
    @outputs M_AMPA2
    @equations begin end
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantDAInput},
    dst::GraphDynamics.Subsystem{MsnD1Receptor},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; DA=c.weight * src.DA))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantDAInput},
    dst::GraphDynamics.Subsystem{MsnD2Receptor},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; DA=c.weight * src.DA))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantGAsympInput},
    dst::GraphDynamics.Subsystem{MsnAMPAR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; G_asymp=c.weight * src.G_asymp))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantMAMPA2Input},
    dst::GraphDynamics.Subsystem{MsnAMPAR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; M_AMPA2=c.weight * src.M_AMPA2))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantModeInput},
    dst::GraphDynamics.Subsystem{HTR5},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; mode=c.weight * src.mode))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantMuscarinicInput},
    dst::GraphDynamics.Subsystem{MuscarinicR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; M=c.weight * src.M))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantVPreInput},
    dst::GraphDynamics.Subsystem{MsnNMDAR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; V_pre=c.weight * src.V_pre))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantVPostInput},
    dst::GraphDynamics.Subsystem{MsnNMDAR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; V_post=c.weight * src.V_post))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantMNMDA1Input},
    dst::GraphDynamics.Subsystem{MsnNMDAR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; M_NMDA1=c.weight * src.M_NMDA1))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantIStimInput},
    dst::GraphDynamics.Subsystem{BaxterSensoryNeuron},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_stim=c.weight * src.I))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantIAppInput},
    dst::GraphDynamics.Subsystem{TRNNeuron},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_app=c.weight * src.I_app))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantIAppInput},
    dst::GraphDynamics.Subsystem{MuscarinicNeuron},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_app=c.weight * src.I_app))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantAChInput},
    dst::GraphDynamics.Subsystem{Alpha7ERnAChR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; ACh=c.weight * src.ACh))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantICaInput},
    dst::GraphDynamics.Subsystem{CaTRPM4R},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_Ca=c.weight * src.I_Ca))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantCaBulkInput},
    dst::GraphDynamics.Subsystem{CaTRPM4R},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; Ca_bulk=c.weight * src.Ca_bulk))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantCChInput},
    dst::GraphDynamics.Subsystem{CaTRPM4R},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; CCh=c.weight * src.CCh))
end

# ConstantNicInput for nicotine concentration
@blox struct ConstantNicInput(; name, namespace=nothing, Nic=0.0) <: AbstractStimulus
    @params Nic
    @states
    @inputs
    @outputs Nic
    @equations begin end
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantAChInput},
    dst::GraphDynamics.Subsystem{Beta2nAChR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; inp_ACh=c.weight * src.ACh))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantNicInput},
    dst::GraphDynamics.Subsystem{Beta2nAChR},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; inp_Nic=c.weight * src.Nic))
end

# VTA neuron support
function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantIAppInput},
    dst::GraphDynamics.Subsystem{VTADANeuron},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_app=c.weight * src.I_app))
end

function (c::BasicConnection)(src::GraphDynamics.Subsystem{ConstantIAppInput},
    dst::GraphDynamics.Subsystem{VTAGABANeuron},
    t)
    acc = GraphDynamics.initialize_input(dst)
    merge(acc, (; I_app=c.weight * src.I_app))
end
