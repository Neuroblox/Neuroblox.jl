using Neuroblox, MAT, ModelingToolkit, OrdinaryDiffEq, Test

"""
Test for learningrate
    Create a vector of behavioral outcomes that are all zeros except for the last window.
    The learning rate should be zero percent for all 1:n-1 windows, and 100% for window n.
"""

outcomes = zeros(100)
outcomes[91:100] .= 1

windows = 10
learning_rate = learningrate(outcomes, windows)

@test sum(learning_rate[1:windows-1]) == 0
@test sum(learning_rate[windows]) == 100

"""
Test for learningblox
"""

## Load Experimental Data
data   = matread("test/phi_p1.mat")
pdata  = data["phi_p1"][1:10000]
fs     = 1000
prange = 0:(1/fs):(length(pdata)-1)*(1/fs)
## Create Cortical and Subcortical Blox
@named cortical    = LearningCircuit(ω=20*(2*pi), d=30.0, 
    prange=vec(prange), pdata=vec(pdata))
@named subcortical = LearningCircuit(ω=20*(2*pi), d=30.0, 
    prange=vec(prange), pdata=vec(pdata))
## Compose Learning Loop
cortical_model    = cortical.odesystem
subcortical_model = subcortical.odesystem
mysys             = compose(cortical_model, subcortical_model)
## Solve System
simsys = ODEProblem(structural_simplify(mysys), [], (0,10.0), [])
sol    = solve(simsys, Rodas3(), dt=0.001, saveat=0.001)

PLV = Neuroblox.PLVTarget(sol[1,:], sol[3,:], 16, 24, 1000, 4)
@test PLV ≈ 1.0


