using Neuroblox, OrdinaryDiffEq, Test

# Create Regions
@named Str = NeuralMass(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = NeuralMass(τ=0.04, H=20, λ=400, r=0.1)
@named STN = NeuralMass(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = NeuralMass(τ=0.014, H=20, λ=400, r=0.1)
@named Th  = NeuralMass(τ=0.002, H=10, λ=20, r=5)
@named EI  = NeuralMass(τ=0.01, H=20, λ=5, r=5)
@named PY  = NeuralMass(τ=0.001, H=20, λ=5, r=0.15)
@named II  = NeuralMass(τ=2.0, H=60, λ=5, r=5)

# Connect Regions through Adjacency Matrix
sys = [Str, GPe, STN, GPi, Th, EI, PY, II]

@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5
adj_matrix = [0 0 0 0 0 0 0 0;
-0.5*C_BG_Th*sys[1].x -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0 0 0 0 0;
              0 -0.5*C_BG_Th*sys[2].x 0 0 0 0 C_Cor_BG_Th*sys[7].x 0;
              0 -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0 0 0 0 0;
              0 0 0 -0.5*C_BG_Th*sys[4].x 0 0 0 0;
              0 0 0 0 C_BG_Th_Cor*sys[5].x 0 6*C_Cor*sys[7].x 0;
              0 0 0 0 0 4.8*C_Cor*sys[6].x 0 -1.5*C_Cor*sys[8].x;
              0 0 0 0 0 0 1.5*C_Cor*sys[7].x 3.3*C_Cor*sys[8].x]


@named CBGTC_Circuit = Connections(sys=sys, adj_matrix=adj_matrix)

sim_dur = 10.0 # Simulate for 10 Seconds
prob = ODAEProblem(structural_simplify(CBGTC_Circuit), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5())
@test sol[GPi.x,4] ≈ 0.9862259241442394
