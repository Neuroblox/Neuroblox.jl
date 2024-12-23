
mutable struct LearningBlox
    ω::Num
    d::Num
    prange::Vector{Float64}
    pdata::Vector{Float64}
    adj::Matrix{Num}
    sys::Vector{ODESystem}
    function LearningBlox(;name, ω=20*(2*pi), d=30, prange=[], pdata=[])
        # Create Blox
        Phase      = PhaseBlox(phase_range=prange, phase_data=pdata,  name=Symbol(String(name)*"_Phase"))
        Cosine     = NoisyCosineBlox(amplitude=1.0, frequency=0.0,    name=Symbol(String(name)*"_Cosine"))
        NeuralMass = HarmonicOscillatorBlox(ω=ω, ζ=1.0, k=(ω)^2, h=d, name=Symbol(String(name)*"_NeuralMass"))
        # Create System
        blox  = [Phase, Cosine, NeuralMass]
        sys   = [s.system for s in blox]
        # Set Internal Connections
        # Columns = Inputs (Sinks); Rows = Outputs (Sources)
               # P C NM
        g     = [0 1 0; # P  
                 0 0 1; # C 
                 0 0 0] # NM      
        adj   = g .* [s.connector for s in blox]
        # Return Properties
        new(ω, d, prange, pdata, adj, sys)
    end
end
