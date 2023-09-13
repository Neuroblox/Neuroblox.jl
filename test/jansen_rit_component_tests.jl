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
sol = simulate(mysys, [], (0.0, sim_dur), [], Vern7(); saveat=0.001)
@test sol[!, "GPi₊x(t)"][4] ≈ -0.48421810231972134

"""
Testing new Jansen-Rit blox

This sets up the exact same system as above, but using the JansenRitBlox with different flags cortical/subcortical flags.

The purpose of this test is to make sure that setting up everything using System rather than ODESystem works as expected.
It also shows test code for the new system_from_graph calls and handling delays when everything is 0 (MethodOfSteps returns the 
same thing as the old simulate call with AutoVern7(Rodas4() since there are no delays.)
"""

# test new Jansen-Rit blox
@named Str = JansenRitBlox(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = JansenRitBlox(τ=0.04, cortical=false) # all default subcortical except τ
@named STN = JansenRitBlox(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = JansenRitBlox(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRitBlox(τ=0.002, H=10, λ=20, r=5)
@named EI  = JansenRitBlox(τ=0.01, H=20, λ=5, r=5)
@named PY  = JansenRitBlox(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRitBlox(τ=2.0, H=60, λ=5, r=5)
blox = [Str, GPe, STN, GPi, Th, EI, PY, II]

# test graphs
g = MetaDiGraph()
add_blox_list!(g, blox)

# Store parameters to be passed later on
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

add_edge!(g, 2, 1, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 2, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 2, 3, Dict(:weight => C_BG_Th))
add_edge!(g, 3, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 3, 7, Dict(:weight => C_Cor_BG_Th))
add_edge!(g, 4, 2, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 4, 3, Dict(:weight => C_BG_Th))
add_edge!(g, 5, 4, Dict(:weight => -0.5*C_BG_Th))
add_edge!(g, 6, 5, Dict(:weight => C_BG_Th_Cor))
add_edge!(g, 6, 7, Dict(:weight => 6*C_Cor))
add_edge!(g, 7, 6, Dict(:weight => 4.8*C_Cor))
add_edge!(g, 7, 8, Dict(:weight => -1.5*C_Cor))
add_edge!(g, 8, 7, Dict(:weight => 1.5*C_Cor))
add_edge!(g, 8, 8, Dict(:weight => 3.3*C_Cor))

# add_edge!(g, 2, 1, Dict(:weight => -0.5*60, :delay => 0.01))
# add_edge!(g, 2, 2, Dict(:weight => -0.5*60, :delay => 0.01))
# add_edge!(g, 2, 3, Dict(:weight => 60, :delay => 0.01))
# add_edge!(g, 3, 2, Dict(:weight => -0.5*60, :delay => 0.01))
# add_edge!(g, 3, 7, Dict(:weight => 5, :delay => 0.01))
# add_edge!(g, 4, 2, Dict(:weight => -0.5*60, :delay => 0.01))
# add_edge!(g, 4, 3, Dict(:weight => 60, :delay => 0.01))
# add_edge!(g, 5, 4, Dict(:weight => -0.5*60, :delay => 0.01))
# add_edge!(g, 6, 5, Dict(:weight => 5, :delay => 0.01))
# add_edge!(g, 6, 7, Dict(:weight => 6*60, :delay => 0.01))
# add_edge!(g, 7, 6, Dict(:weight => 4.8*60, :delay => 0.01))
# add_edge!(g, 7, 8, Dict(:weight => -1.5*60, :delay => 0.01))
# add_edge!(g, 8, 7, Dict(:weight => 1.5*60, :delay => 0.01))
# add_edge!(g, 8, 8, Dict(:weight => 3.3*60, :delay => 0.01))

(final_system, final_delays) = system_from_graph(g, params, true; name=:final_system)
sim_dur = 10.0 # Simulate for 10 Seconds
final_system_sys = structural_simplify(final_system)
prob = DDEProblem(final_system_sys,
    [],
    (0.0, sim_dur),
    constant_lags = final_delays)
alg = MethodOfSteps(Vern7())
sol_dde_no_delays = solve(prob, alg, saveat=0.001)
sol2 = DataFrame(sol_dde_no_delays)
@test sol2[!, "GPi₊x(t)"][4] ≈ -0.48421810231972134