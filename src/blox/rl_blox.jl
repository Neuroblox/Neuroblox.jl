
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