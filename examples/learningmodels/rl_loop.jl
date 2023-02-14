## Load Packages
using Neuroblox, MAT, ModelingToolkit, OrdinaryDiffEq, Plots


## Load Experimental Data (PFC)
data       = matread("examples/learningmodels/phi_p1.mat")
fs         = 1000
pdata_PFC  = data["phi_p1"][1:5000]
prange_PFC = 0:(1/fs):(length(pdata_PFC)-1)*(1/fs)
dataset_PFC = [prange_PFC, pdata_PFC]
## Declare Parameters (PFC)
ω_PFC = 20*(2*pi)
d_PFC = 30
## Load Experimental Data (STR)
data       = matread("examples/learningmodels/phi_s1.mat")
fs         = 1000
pdata_STR  = data["phi_s1"][1:5000]        
prange_STR = 0:(1/fs):(length(pdata_STR)-1)*(1/fs)
dataset_STR = [prange_STR, pdata_STR]
## Declare Parameters (STR)
ω_STR = 20*(2*pi)
d_STR = 30



## LearningBlox Struct
@parameters t
mutable struct LearningBlox
    ω::Num
    d::Num
    prange::Vector{Float64}
    pdata::Vector{Float64}
    ROI::String
    adj::Matrix{Num}
    sys::Vector{ODESystem}
    function LearningBlox(;name, ω=20*(2*pi), d=30, prange=[], pdata=[], ROI="")
        # Create Blox
        Phase      = PhaseBlox(phase_range=prange, phase_data=pdata,  name=Symbol("Phase"*ROI))
        Cosine     = NoisyCosineBlox(amplitude=1.0, frequency=0.0,    name=Symbol("Cosine"*ROI))
        NeuralMass = HarmonicOscillatorBlox(ω=ω, ζ=1.0, k=(ω)^2, h=d, name=Symbol("NeuralMass"*ROI))
        # Create System
        blox  = [Phase, Cosine, NeuralMass]
        sys   = [s.odesystem for s in blox]
        # Set Internal Connections
        g     = [0 1 0; 0 0 1; 0 0 0]
        adj   = g .* [s.connector for s in blox]
        # Return Properties
        new(ω, d, prange, pdata, ROI, adj, sys)
    end
end


## Create Learning Loop
function create_rl_loop(;name, ROIs, datasets, parameters, c_ext)
    # Create LearningBlox for each Region
    regions = []
    for r in eachindex(ROIs)
        push!(regions, 
            LearningBlox(
                ω=parameters[:ω][r], d=parameters[:d][r], 
                prange=vec(datasets[r][1]), pdata=vec(datasets[r][2]), 
                ROI=ROIs[r], name=Symbol(ROIs[r])
            )
        )
    end
    # Connect Regions through an External Connection Weight
    @parameters c_ext=c_ext
    for r in eachindex(ROIs)
        regions[r].adj[3,3] = c_ext*regions[1:end .!= r, :][1].sys[3].x
    end
    # Update Adjacency Matrix to Incorporate External Connections
    eqs = []
    for r in eachindex(ROIs)
        for s in eachindex(regions[r].sys) 
            push!(eqs, regions[r].sys[s].jcn ~ sum(regions[r].adj[:, s]))
        end
    end
    # Compose Loop
    sys = []
    for r in eachindex(ROIs)
        sys = vcat(sys, regions[r].sys)
    end
    # Return One ODESystem
    return ODESystem(eqs, systems=sys, name=name)
end

# Create Model
@named corticostriatal_loop = rl_loop(
    ROIs       = ["PFC", "STR"],
    parameters = Dict(:ω => (ω_PFC, ω_STR), :d => (d_PFC, d_STR)),
    datasets   = [dataset_PFC, dataset_STR],
    c_ext      = 0.04
    )

# Solve System 
prob = ODEProblem(structural_simplify(corticostriatal_loop), [], (0, 5.0), [])
sol = solve(prob, Rodas3(), saveat=0.001, dt=0.001) 
# Plot Solution
plot(sol.t,  sol[1,:], label="NeuralMassPFC.x", lw=:1.6, lc=:blue)
plot!(sol.t, sol[3,:], label="NeuralMassSTR.x", lw=:1.6, lc=:orange)
xlabel!("seconds")
ylabel!("amplitude")
title!("Simulated LFP")