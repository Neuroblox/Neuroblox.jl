# [Getting Started with Neuroblox](@id neuroblox_example)

This tutorial will introduce you to simulating brain dynamics using Neuroblox.

## Example 1 : Building an oscillating circuit from two Wilson-Cowan Neural Mass Models

```@example Wilson-Cowan
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

@named WC1 = WilsonCowan()
@named WC2 = WilsonCowan()

adj = [-1 6; 6 -1]
g = MetaDiGraph()
add_blox!.(Ref(g), [WC1, WC2])
create_adjacency_edges!(g, adj)

@named sys = system_from_graph(g)
sys = structural_simplify(sys)

prob = ODEProblem(sys, [], (0.0, 100), [])
sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)
plot(sol)
```

## Example 2 : Building a Brain Circuit using Neural Mass Models

In this example, we will construct a Parkinsons model from eight Jansen-Rit Neural Mass Models.  The Jansen-Rit Neural Mass model is defined by the following differential equations:

```math
\\frac{dx}{dt} = y-\\frac{2}{\\tau}x
\\frac{dy}{dt} = -\\frac{x}{\\tau^2} + \\frac{H}{\\tau} [\\frac{2\\lambda}{1+\\text{exp}(-r*\\sum{jcn})} - \\lambda]
```

```@example Jansen-Rit
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

@named str = JansenRit(τ=2.2, H=20, λ=300, r=0.3)
@named gpe = JansenRit(τ=40.0, cortical=false) # all default subcortical except τ
@named stn = JansenRit(τ=10.0, H=20, λ=500, r=0.1)
@named gpi = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=2.0, H=10, λ=20, r=5)
@named EI  = JansenRit(τ=10.0, H=20, λ=5, r=5)
@named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2000.0, H=60, λ=5, r=5)
blox = [str, gpe, stn, gpi, Th, EI, PY, II]

# Store parameters to be passed later on
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 0 0 0 0 0 0 0;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
            0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

g = MetaDiGraph()
add_blox!.(Ref(g), blox)
create_adjacency_edges!(g, adj_matrix_lin)

@named final_system = system_from_graph(g, params)

sim_dur = 2000.0 # Simulate for 2 Seconds
final_system_sys = structural_simplify(final_system)
prob = ODEProblem(final_system_sys, [], (0.0, sim_dur),[])

sol = solve(prob, Tsit5(), saveat=1)
#plot(sol)
```
