using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

"""
Tests formerly in components.jl
"""
# Create Regions
@named Str = jansen_ritC(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
@named STN = jansen_ritC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)
@named Th  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
@named EI  = jansen_ritSC(τ=0.01, H=20, λ=5, r=5)
@named PY  = jansen_ritSC(τ=0.001, H=20, λ=5, r=0.15)
@named II  = jansen_ritSC(τ=2.0, H=60, λ=5, r=5)

# Connect Regions through Adjacency Matrix
blox = [Str, GPe, STN, GPi, Th, EI, PY, II]
sys = [s.odesystem for s in blox]
connect = [s.connector for s in blox]

@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 0 0 0 0 0 0 0;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
            0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

@named CBGTC_Circuit_lin = LinearConnections(sys=sys, adj_matrix=adj_matrix_lin, connector=connect)

sim_dur = 10.0 # Simulate for 10 Seconds
mysys = structural_simplify(CBGTC_Circuit_lin)
sol = simulate(mysys, [], (0.0, sim_dur), [], Vern7(); saveat=0.001)
@test sol[!, "GPi₊x(t)"][4] ≈ -0.48421810231972134

sol = simulate(mysys, random_initials(mysys,blox),(0.0, sim_dur), [])
@test size(sol)[2] == 17 # make sure that all the states are simulated (16 + timestamp)

"""
New tests for JansenRit blox

These are to make sure the new JansenRit blox works identically to the former one.
"""

# see jansen_rit_component_tests.jl