# A basic example using delayed differential equations to introduce optional delays in interregional connections.
# The system being built here is the same as in Liu et al. (2020). DOI: 10.1016/j.neunet.2019.12.021.

using Neuroblox  # Core functionality
using DifferentialEquations # Needed for solver
using MetaGraphs # Set up graph of systems

τ_factor = 1000 #needed because the paper units were in seconds, and we need ms to be consistent

# Create Jansen-Rit blocks with the same parameters as the paper and store them in a list
@named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named GPE = JansenRit(τ=0.04*τ_factor, cortical=false) # all default subcortical except τ
@named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1)
@named GPI = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5)
@named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5)
blox = [Str, GPE, STN, GPI, Th, EI, PY, II]

# test graphs
g = MetaDiGraph()
add_blox!.(Ref(g), blox)

# Store parameters to be passed later on
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

# Add the edges as specified in Table 2 of Liu et al.
# This is a subset of the edges selected to run a shorter version of the model.
# If you want to add more edges in a batch, you can create an adjacency matrix and then call create_adjacency_edges!(g, adj_matrix).
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

# Create the ODE system. This will have some warnings as delays are set to 0 - ignore those for now.
@named final_system = system_from_graph(g, params)
final_system_sys = structural_simplify(final_system)

# Collect the graph delays and create a DDEProblem. This will all be zero in this case.
final_delays = graph_delays(g)
sim_dur = 1000.0 # Simulate for 1 second
prob = ODEProblem(final_system_sys,
    [],
    (0.0, sim_dur))

# Select the algorihm. MethodOfSteps will return the same as Vern7() in this case because there are no non-zero delays, but is required since this is a DDEProblem.
alg = Vern7()
sol_dde_no_delays = solve(prob, alg, saveat=1)


# Example of delayed connections

# First, recreate the graph to remove previous connections
@named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3, delayed=true)
@named GPE = JansenRit(τ=0.04*τ_factor, cortical=false, delayed=true) # all default subcortical except τ
@named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1, delayed=true)
@named GPI = JansenRit(cortical=false, delayed=true) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5, delayed=true)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5, delayed=true)
@named PY  = JansenRit(cortical=true, delayed=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5, delayed=true)
blox = [Str, GPE, STN, GPI, Th, EI, PY, II]
g = MetaDiGraph()
add_blox!.(Ref(g), blox)

# Now, add the edges with the specified delays. Again, if you prefer, there's a version using adjacency and delay matrices to assign all at once.
add_edge!(g, 2, 1, Dict(:weight => -0.5*60, :delay => 1))
add_edge!(g, 2, 2, Dict(:weight => -0.5*60, :delay => 2))
add_edge!(g, 2, 3, Dict(:weight => 60, :delay => 1))
add_edge!(g, 3, 2, Dict(:weight => -0.5*60, :delay => 1))
add_edge!(g, 3, 7, Dict(:weight => 5, :delay => 1))
add_edge!(g, 4, 2, Dict(:weight => -0.5*60, :delay => 4))
add_edge!(g, 4, 3, Dict(:weight => 60, :delay => 1))
add_edge!(g, 5, 4, Dict(:weight => -0.5*60, :delay => 2))
add_edge!(g, 6, 5, Dict(:weight => 5, :delay => 1))
add_edge!(g, 6, 7, Dict(:weight => 6*60, :delay => 2))
add_edge!(g, 7, 6, Dict(:weight => 4.8*60, :delay => 3))
add_edge!(g, 7, 8, Dict(:weight => -1.5*60, :delay => 1))
add_edge!(g, 8, 7, Dict(:weight => 1.5*60, :delay => 4))
add_edge!(g, 8, 8, Dict(:weight => 3.3*60, :delay => 1))

# Now you can run the same code as above, but it will handle the delays automatically.
@named final_system = system_from_graph(g)
final_system_sys = structural_simplify(final_system)

# Collect the graph delays and create a DDEProblem.
final_delays = graph_delays(g)
sim_dur = 1000.0 # Simulate for 1 second
prob = DDEProblem(final_system_sys,
    [],
    (0.0, sim_dur),
    constant_lags = final_delays)

# Select the algorihm. MethodOfSteps is now needed because there are non-zero delays.
alg = MethodOfSteps(Vern7())
sol_dde_with_delays = solve(prob, alg, saveat=1)