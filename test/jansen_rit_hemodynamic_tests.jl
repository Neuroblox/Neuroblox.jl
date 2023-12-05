using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

# Store parameters to be passed later on
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 0 0 0 0 0 0 0 1;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor 0;
            0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor 0;
            0 0 0 0 0 0 0 0 0]

# test new Jansen-Rit blox
@named Str = JansenRit(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = JansenRit(τ=0.04, cortical=false) # all default subcortical except τ
@named STN = JansenRit(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002, H=10, λ=20, r=5)
@named EI  = JansenRit(τ=0.01, H=20, λ=5, r=5)
@named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0, H=60, λ=5, r=5)
@named hemo = AlternativeBalloonModel()
blox = [Str, GPe, STN, GPi, Th, EI, PY, II, hemo]

# Alternative version using adjacency matrix
g = MetaDiGraph()
add_blox!.(Ref(g), blox)
create_adjacency_edges!(g, adj_matrix_lin)

@named final_system = system_from_graph(g, params)
final_delays = graph_delays(g)
sim_dur = 600.0 # Simulate for 10 Seconds
final_system_sys = structural_simplify(final_system)
prob = DDEProblem(final_system_sys,
    [],
    (0.0, sim_dur),
    constant_lags = final_delays)
alg = MethodOfSteps(Vern7())
sol_dde_no_delays = solve(prob, alg, saveat=0.001)
sol3 = DataFrame(sol_dde_no_delays)