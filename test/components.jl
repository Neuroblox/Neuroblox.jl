using Neuroblox, OrdinaryDiffEq, DataFrames, Test, Distributions

"""
neuralmass.jl test
"""

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

@named CBGTC_Circuit_lin = LinearConnections(sys=sys, adj_matrix=adj_matrix_lin, connector=[s.x for s in sys])

sim_dur = 10.0 # Simulate for 10 Seconds
sol = simulate(CBGTC_Circuit_lin, [], (0.0, sim_dur), [])
@test sol[!,"GPi₊x(t)"][4] ≈ 0.9785615009584057

"""
thetaneuron.jl test

Test approach: generate a network of theta neurons and connect them through an all-to-all
adjacency matrix via a spike function. Then compute the real part of the Kuramoto order parameter.
The average of this parameter should be close to zero as synchrony varies in the network from a positive
to a negative amplitude. 
"""

# Generate Theta Network
network = [] 
N = 500
for i = 1:N
    η  = rand(Cauchy(1.0, 0.05)) # Constant Drive
    @named neuron = Neuroblox.ThetaNeuron(name=Symbol("neuron$i"), η=η, α_inv=1.0, k=-2.0, N=500)
    push!(network, neuron)
end

# Create Circuit
adj_matrix = ones(N,N)
n = 3
a_n = 2.0^n*(factorial(n)^2.0)/(factorial(2.0*n))
@named theta_circuit = LinearConnections(sys=network, adj_matrix=adj_matrix, connector=[a_n*(1-cos(neuron.θ))^n for neuron in network])

sim_dur = 50.0 # Simulate for 10 Seconds
sol = simulate(theta_circuit, [], (0.0, sim_dur), [])
R = real(exp.(im*sol[!, "neuron1₊θ(t)"]))
@test mean(R) < 0.1
@test mean(R) > -0.1