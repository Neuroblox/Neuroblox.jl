using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

τ_factor = 1000 #needed because the old units were in seconds, and we need ms to be consistent

# Create Regions
@named Str = jansen_ritC(τ=0.0022*τ_factor, H=20, λ=300, r=0.3)
@named gpe = jansen_ritC(τ=0.04*τ_factor, H=20, λ=400, r=0.1)
@named stn = jansen_ritC(τ=0.01*τ_factor, H=20, λ=500, r=0.1)
@named gpi = jansen_ritSC(τ=0.014*τ_factor, H=20, λ=400, r=0.1)
@named Th  = jansen_ritSC(τ=0.002*τ_factor, H=10, λ=20, r=5)
@named EI  = jansen_ritSC(τ=0.01*τ_factor, H=20, λ=5, r=5)
@named PY  = jansen_ritSC(τ=0.001*τ_factor, H=20, λ=5, r=0.15)
@named II  = jansen_ritSC(τ=2.0*τ_factor, H=60, λ=5, r=5)

# Connect Regions through Adjacency Matrix
blox = [Str, gpe, stn, gpi, Th, EI, PY, II]
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

sim_dur = 2000.0 # Simulate for 2 seconds
mysys = structural_simplify(CBGTC_Circuit_lin)
sol = simulate(mysys, [], (0.0, sim_dur), [], Vern7(); saveat=1)
@test sol[!, "gpi₊x(t)"][4] ≈ -2219.2560209502685 #updated to new value in ms

"""
Testing new Jansen-Rit blox

This sets up the exact same system as above, but using the JansenRit with different flags cortical/subcortical flags.

The purpose of this test is to make sure that setting up everything using System rather than ODESystem works as expected.
It also shows test code for the new system_from_graph calls and handling delays when everything is 0 (MethodOfSteps returns the 
same thing as the old simulate call with AutoVern7(Rodas4() since there are no delays.)
"""

# test new Jansen-Rit blox
@named Str = JansenRit(τ=0.0022*τ_factor, H=20, λ=300, r=0.3)
@named gpe = JansenRit(τ=0.04*τ_factor, cortical=false) # all default subcortical except τ
@named stn = JansenRit(τ=0.01*τ_factor, H=20, λ=500, r=0.1)
@named gpi = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10, λ=20, r=5)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20, λ=5, r=5)
@named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60, λ=5, r=5)
blox = [Str, gpe, stn, gpi, Th, EI, PY, II]

# test graphs
g = MetaDiGraph()
add_blox!.(Ref(g), blox)

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

@named final_system = system_from_graph(g, params)
final_delays = graph_delays(g)
sim_dur = 2000.0 # Simulate for 2 Seconds
final_system_sys = structural_simplify(final_system)
prob = ODEProblem(final_system_sys,
    [],
    (0.0, sim_dur))
alg = Vern7()
sol_dde_no_delays = solve(prob, alg, saveat=1)
sol2 = DataFrame(sol_dde_no_delays)
@test isapprox(sol2[!, "gpi₊x(t)"][500:1000], sol[!, "gpi₊x(t)"][500:1000], rtol=1e-8)


# Alternative version using adjacency matrix
g2 = MetaDiGraph()
add_blox!.(Ref(g2), blox)
create_adjacency_edges!(g2, adj_matrix_lin)

@named final_system = system_from_graph(g2, params)
final_delays = graph_delays(g2)
sim_dur = 2000.0 # Simulate for 2 Seconds
final_system_sys = structural_simplify(final_system)
prob = DDEProblem(final_system_sys,
    [],
    (0.0, sim_dur),
    constant_lags = final_delays)
alg = MethodOfSteps(Vern7())
sol_dde_no_delays = solve(prob, alg, saveat=1)
sol3 = DataFrame(sol_dde_no_delays)
@test isapprox(sol3[!, "gpi₊x(t)"][500:1000], sol[!, "gpi₊x(t)"][500:1000], rtol=1e-8)