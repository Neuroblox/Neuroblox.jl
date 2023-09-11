using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

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
sol = simulate(mysys, [], (0.0, sim_dur), [])
@test sol[!, "GPi₊x(t)"][4] ≈ 0.976257006970988

# test blox
# @named PY  = JansenRitBlox(τ=0.001, H=20, λ=5, r=0.15)
# @named EI  = JansenRitBlox(τ=0.01, H=20, λ=5, r=5)
# @named II  = JansenRitBlox(τ=2.0, H=60, λ=5, r=5)

# # test graphs
g = MetaDiGraph()
add_blox!(g, PY)
add_blox!(g, EI)
add_blox!(g, II)
add_edge!(g, 1, 2, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 2, 3, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 3, 1, Dict(:weight => 1.0, :delay => 1.5))

# @named final_system = system_from_graph(g)
# sim_dur = 10.0 # Simulate for 10 Seconds
# sys = structural_simplify(final_system)
# prob = DDEProblem(sys,
#     [],
#     (0.0, sim_dur),
#     constant_lags = [1])
# alg = MethodOfSteps(Tsit5())
# sol_mtk = solve(prob, alg, reltol = 1e-7, abstol = 1e-10, saveat=0.001)