using Neuroblox, OrdinaryDiffEq, Test

function CreateCircuit(;name)

       # Create Regions
       @named Str = NeuralMass(τ=0.0022, H=20, λ=300, r=0.3)
       @named GPe = NeuralMass(τ=0.04, H=20, λ=400, r=0.1)
       @named STN = NeuralMass(τ=0.01, H=20, λ=500, r=0.1)
       @named GPi = NeuralMass(τ=0.014, H=20, λ=400, r=0.1)
       @named Th  = NeuralMass(τ=0.002, H=10, λ=20, r=5)
       @named EI  = NeuralMass(τ=0.01, H=20, λ=5, r=5)
       @named PY  = NeuralMass(τ=0.001, H=20, λ=5, r=0.15)
       @named II  = NeuralMass(τ=2.0, H=60, λ=5, r=5)

       # Connect Regions
       sys = [Str, GPe, STN, GPi, Th, EI, PY, II]
       @named Circuit = Connections(sys=sys)

end
@named CBGTC_Circuit = CreateCircuit()

sim_dur = 10.0 # Simulate for 10 Seconds
prob = ODAEProblem(structural_simplify(CBGTC_Circuit), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5())
@test sol[end][4] ≈ -3977.947882200623
