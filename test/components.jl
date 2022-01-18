using Neuroblox, OrdinaryDiffEq, Test

# Create Regions
@named Str = NeuralMass(activation="logistic", τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = NeuralMass(activation="logistic", τ=0.04, H=20, λ=400, r=0.1)
@named STN = NeuralMass(activation="logistic", τ=0.01, H=20, λ=500, r=0.1)
@named GPi = NeuralMass(activation="logistic", τ=0.014, H=20, λ=400, r=0.1)
@named Th  = NeuralMass(activation="logistic", τ=0.002, H=10, λ=20, r=5)
@named EI  = NeuralMass(activation="logistic", τ=0.01, H=20, λ=5, r=5)
@named PY  = NeuralMass(activation="logistic", τ=0.001, H=20, λ=5, r=0.15)
@named II  = NeuralMass(activation="logistic", τ=2.0, H=60, λ=5, r=5)

# Connect Regions through Adjacency Matrix
sys = [Str, GPe, STN, GPi, Th, EI, PY, II]

@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 0 0 0 0 0 0 0;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
            0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

@named CBGTC_Circuit_lin = LinearConnections(sys=sys, adj_matrix=adj_matrix_lin)

sim_dur = 10.0 # Simulate for 10 Seconds
prob = ODAEProblem(structural_simplify(CBGTC_Circuit_lin), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5())
@test_broken sol[GPi.x,4] ≈ 0.9862259241442394
@test sol[GPi.x,4] ≈ 0.9785615009584057
