
@parameters t

mutable struct LearningCircuit
    ω::Num
    d::Num
    prange::Vector{Float64}
    pdata::Vector{Float64}
    connector::Matrix{Num}
    odesystem::ODESystem
    function LearningBlox(;name, ω=20*(2*pi), d=30, prange=[], pdata=[])

        @named Internal_Phase  = PhaseBlox(phase_range=prange, phase_data=pdata)
        @named Internal_Input  = NoisyCosineBlox(amplitude=1.0, frequency=0.0)
        @named Neural_Model    = harmonic_oscillator(ω=ω, ζ=1.0, k=(ω)^2, h=d)

        blox    = [Internal_Phase, Internal_Input, Neural_Model]
        sys     = [s.odesystem for s in blox]

        g       = [0 1 0;
                   0 0 1;
                   0 0 0]  
        connect = [s.connector for s in blox]
        adj     = g .* connect

        eqs = []
        for rg in eachindex(sys) 
           push!(eqs, sys[rg].jcn ~ sum(adj[:, rg]))
        end
        odesys = ODESystem(eqs, name=name, systems=sys)

        new(ω, d, prange, pdata, adj, odesys)
    end
end