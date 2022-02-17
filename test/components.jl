using Neuroblox, OrdinaryDiffEq, DataFrames, Test, Distributions, Statistics, LinearAlgebra


"""
neuralmass.jl test
"""

# Create Regions
@named Str = neuralmass(activation="logistic", τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = neuralmass(activation="logistic", τ=0.04, H=20, λ=400, r=0.1)
@named STN = neuralmass(activation="logistic", τ=0.01, H=20, λ=500, r=0.1)
@named GPi = neuralmass(activation="logistic", τ=0.014, H=20, λ=400, r=0.1)
@named Th  = neuralmass(activation="logistic", τ=0.002, H=10, λ=20, r=5)
@named EI  = neuralmass(activation="logistic", τ=0.01, H=20, λ=5, r=5)
@named PY  = neuralmass(activation="logistic", τ=0.001, H=20, λ=5, r=0.15)
@named II  = neuralmass(activation="logistic", τ=2.0, H=60, λ=5, r=5)

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
    @named neuron = Neuroblox.thetaneuron(name=Symbol("neuron$i"), η=η, α_inv=1.0, k=-2.0)
    push!(network, neuron)
end

# Create Circuit
adj_matrix = (1/N)*ones(N,N)
n = 3
a_n = 2.0^n*(factorial(n)^2.0)/(factorial(2.0*n))
@named theta_circuit = LinearConnections(sys=network, adj_matrix=adj_matrix, connector=[a_n*(1-cos(neuron.θ))^n for neuron in network])

sim_dur = 50.0 # Simulate for 10 Seconds
sol = Neuroblox.simulate(theta_circuit, [], (0.0, sim_dur), [])
R = real(exp.(im*sol[!, "neuron1₊θ(t)"]))

@test Statistics.mean(R) < 0.6
@test Statistics.mean(R) > -0.6


"""
qif_neuron.jl and synaptic_network.jl test

This test generates a network of quadratic integrate and fire neurons 
using qif_neuron.jl and connects them with synapses using synaptic_network.jl
This should successfully generate a structurally simplified ODESystem for 
the entire network. If N is number of neurons and S+1 is the number of state variables
for each neuron (S internal variables and 1 synaptic input), then the total number of states
for the resulting ODESystem for network should be N*S.
"""

#Generate qif neurons
N_nrn = 10	
nrn_network=[]

ω₀ = 0.269
Δω = 0.042

for ii = 1:N_nrn
	 nn = qif_neuron(name=Symbol("nrn$ii"),C=30.0,E_syn=-10,G_syn=1,ω=rand(Cauchy(ω₀,Δω)),τ=35)
     push!(nrn_network,nn)
end

# create synaptic network
k = 0.105 #synaptic weight
adj = ones(N_nrn,N_nrn)
for ii = 1:N_nrn
	adj[ii,ii]=0
end
syn = adj.*k/N_nrn

@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn)

@test typeof(syn_net) == ODESystem
@test length(states(syn_net)) == 3*N_nrn

#simulate for 100 s
sim_dur =  100.0
sol = simulate_neurons(syn_net, [], (0.0, sim_dur), [], Rodas5())

@test sol[end,1] == sim_dur

"""
complex neural mass model test (next generation neural mass model)
This test generates a neural mass model using the kuramoto order parameter
to capture within-population synchrony. A model is generated and then
the phase of oscillations is computed (ψ) along with synchrony (R). 
This model has no input, and therefore oscillations and synchrony should
tend toward zero.
"""
@named macroscopic_model = neuralmass(activation="complex", C=30, Δ=1.0, η_0=5.0, v_syn=-10, alpha_inv=35, k=0.105)
sim_dur = 1000.0 
sol = simulate_complex(macroscopic_model, [], (0.0, sim_dur), [], Tsit5())

C=30
W = (1 .- conj.(sol[1,:]))./(1 .+ conj.(sol[1,:]))
R = (1/(C*pi))*(W+conj.(W))/2
ψ = log.(sol[1,:]./R)/im

@test norm.(R[length(R)]) < 0.1