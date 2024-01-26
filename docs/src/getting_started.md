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
\frac{dx}{dt} = y-\frac{2}{\tau}x
\frac{dy}{dt} = -\frac{x}{\tau^2} + \frac{H}{\tau} [\frac{2\lambda}{1+\text{exp}(-r*\sum{jcn})} - \lambda]
```

```@example Jansen-Rit
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

@named GPe       = JansenRit(τ=40.0, H=20, λ=400, r=0.1)
@named STN       = JansenRit(τ=10.0, H=20, λ=500, r=0.1)
@named GPi       = JansenRit(τ=14.0, H=20, λ=400, r=0.1)
@named Thalamus  = JansenRit(τ=2.0, H=10, λ=20, r=5)
@named PFC       = JansenRit(τ=1.0, H=20, λ=5, r=0.15)
blox = [GPe, STN, GPi, Thalamus, PFC]

# Store parameters to be passed later on
params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

g = MetaDiGraph()
add_blox!.(Ref(g), blox)

add_edge!(g,1,1,:weight, -0.5*C_BG_Th)
add_edge!(g,1,2,:weight, C_BG_Th)
add_edge!(g,2,1,:weight, -0.5*C_BG_Th)
add_edge!(g,2,5,:weight, C_Cor_BG_Th)
add_edge!(g,3,1,:weight, -0.5*C_BG_Th)
add_edge!(g,3,2,:weight, C_BG_Th)
add_edge!(g,4,3,:weight, -0.5*C_BG_Th)
add_edge!(g,4,4,:weight, C_BG_Th_Cor)

@named sys = system_from_graph(g, params)
sys_delays = graph_delays(g)
sim_dur = 2000.0 # Simulate for 2 Seconds
sys = structural_simplify(sys)

# Jansen-Rit allows delays and therefore we create a delayed
# differential equation problem
prob = DDEProblem(sys,
    [],
    (0.0, sim_dur),
    constant_lags = sys_delays)
alg = MethodOfSteps(Vern7())
sol_dde_no_delays = solve(prob, alg, saveat=1)

plot(sol_dde_no_delays)
```
