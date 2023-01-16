using Neuroblox, ModelingToolkit, OrdinaryDiffEq, MAT, DataInterpolations, Plots

## Create a Cortico-Striatal Loop
function LearningBlox(ω_c, ω_sc, d_c, d_sc, prange_c, pdata_c, prange_sc, pdata_sc)

   # Generate Neural Regions
   function CorticalRegion()
        @named Model_c = harmonic_oscillator(ω=ω_c, ζ=1.0, k=(ω_c)^2, h=d_c)
        @named Input_c = NoisyCosineBlox(amplitude=1, frequency=0)
        @named Noise_c = PhaseBlox(phase_range=prange_c, phase_data=pdata_c)
        return [Noise_c, Input_c, Model_c]
    end
    blox_c  = CorticalRegion()

    function SubCorticalRegion()
        @named Model_sc = harmonic_oscillator(ω=ω_sc, ζ=1.0, k=(ω_sc)^2, h=d_sc)
        @named Input_sc = NoisyCosineBlox(amplitude=1, frequency=0)
        @named Noise_sc = PhaseBlox(phase_range=prange_sc, phase_data=pdata_sc)
        return [Noise_sc, Input_sc, Model_sc]
    end
    blox_sc = SubCorticalRegion()

    # Connect Components
    blox    = append!(blox_c, blox_sc)
    sys     = [s.odesystem for s in blox]
    connect = [s.connector for s in blox]
    g       = [0 1 0 0 0 0;
               0 0 1 0 0 1;
               0 0 0 0 0 0
               0 0 0 0 1 0
               0 0 0 0 0 1
               0 0 1 0 0 0]

    # Generate ODE System
    @named Circuit = LinearConnections(sys=sys, adj_matrix=g, connector=connect)    
    return structural_simplify(Circuit)
end

## Load Experimental Data
fs       = 1000
dt       = 1/fs
# Cortical
data     = matread("examples/learningmodels/phi_p1.mat")
pdata_c  = data["phi_p1"][1:20000]
prange_c = 0:dt:(length(pdata_c)-1)*dt
pdata_c  = convert(Vector{Float64}, pdata_c)
prange_c = convert(Vector{Float64}, prange_c)
# SubCortical
data     = matread("examples/learningmodels/phi_s1.mat")
pdata_sc = data["phi_s1"][1:20000]
prange_sc = 0:dt:(length(pdata_sc)-1)*dt
pdata_sc  = convert(Vector{Float64}, pdata_sc)
prange_sc = convert(Vector{Float64}, prange_sc)

## Create Control Circuit
Control_Circuit = LearningBlox(20*(2*pi), 20*(2*pi), 30.0, 30.0, prange_c, pdata_c, prange_sc, pdata_sc)
## Solve the System
System = ODEProblem(Control_Circuit, [], (0,20.0), [])
sol = solve(System, Rodas3(), dt=0.001, saveat=0.001)


figure = @layout [a;b]

figa = plot(sol.t, sol[1,:],  label="Cortical Blox", lw=:1.5)
figa = plot!(sol.t, sol[3,:], label="SubCortical Blox", lw=:1.5,
title="Simulated LFP Time Series", xlabel="Session Time (s)", ylabel="mV^2/Hz")

controller_call_time = Int(0.8/dt)
PLV = []
for t in 1:length(sol.t)

    if rem(t, controller_call_time)==0
        push!(PLV, Neuroblox.PLVTarget(
            sol[1,t-controller_call_time+1:t], sol[3,t-controller_call_time+1:t], 
            16, 24, 1000, 4))
    end
end
figb = plot(PLV, label="PLV", title="Simulated Phase-Locking Value", 
xlabel="Trial Number", ylabel="PLV", xticks=(1:1:30), lc=:black, lw=:3.0, ylims=(0.5, 1.0))

plot(figa, figb, layout=figure, size=[800,800])
