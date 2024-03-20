# [Getting Started with Neuroblox](@id neuroblox_example)

This tutorial will introduce you to simulating brain dynamics using Neuroblox.

## Example 1 : Building an oscillating circuit from two Wilson-Cowan Neural Mass Models

The Wilson–Cowan model describes the dynamics of interactions between populations of excitatory and inhibitory neurons. Each Wilson-Cowan Blox is described by the follwoing equations:

```math
\frac{dE}{dt} = \frac{-E}{\tau_E} + \frac{1}{1 + \text{exp}(-a_E*(c_{EE}*E - c_{IE}*I - \theta_E + \eta*(\sum{jcn}))}\\[10pt]
\frac{dI}{dt} = \frac{-I}{\tau_I} + \frac{1}{1 + exp(-a_I*(c_{EI}*E - c_{II}*I - \theta_I)}
```

Our first example is to simply combine two Wilson-Cowan Blox to build an oscillatory circuit

```@example Wilson-Cowan
using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

@named WC1 = WilsonCowan()
@named WC2 = WilsonCowan()

g = MetaDiGraph()
add_blox!.(Ref(g), [WC1, WC2])

adj = [-1 6; 6 -1]
create_adjacency_edges!(g, adj)

```

First, we create the two Wilson-Cowan Blox: WC1 and WC2. Next, we add the two Blox into a directed graph as nodes and then we are creating weighted edges between the two nodes using an adjacency matrix.

Now we are ready to build the ModelingToolkit System.  Structural simplify creates the final set of equations in which all substiutions are made.

```@example Wilson-Cowan
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
```

To solve the system, we first create an ODEProblem and then solve it over the tspan of (0,100) using a stiff solver.  The solution is saved every 0.1ms. The unit of time in Neuroblox is 1ms.

```@example Wilson-Cowan
prob = ODEProblem(sys, [], (0.0, 100), [])
sol = solve(prob, Rodas4(), saveat=0.1)
plot(sol)
```

## Example 2 : Building a Brain Circuit from literature using Neural Mass Models

In this example, we will construct a Parkinsons model from eight Jansen-Rit Neural Mass Models as described in Liu et al. (2020). DOI: 10.1016/j.neunet.2019.12.021. The Jansen-Rit Neural Mass model is defined by the following differential equations:

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
```
The original paper units are in seconds we therefore need to multiply all parameters with a common factor

```@example Jansen-Rit
τ_factor = 1000

@named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named GPE = JansenRit(τ=0.04*τ_factor, cortical=false) # all default subcortical except τ
@named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1)
@named GPI = JansenRit(cortical=false) # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5)
@named PY  = JansenRit(cortical=true) # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5)
blox = [Str, GPE, STN, GPI, Th, EI, PY, II]
```
Again, we create a graph and add the Blox as nodes

```@example Jansen-Rit
g = MetaDiGraph()
add_blox!.(Ref(g), blox)

params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5
```
ModelingToolkit allows us to create parameters that can be passed into the equations symbolically.

We add edges as specified in Table 2 of Liu et al.
We only implemented a subset of the nodes and edges to describe a less complex version of the model. Edges can also be created using an adjacency matrix as in the previous example.

```@example Jansen-Rit
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
add_edge!(g,1,1,:weight, -0.5*C_BG_Th)
add_edge!(g,1,2,:weight, C_BG_Th)
add_edge!(g,2,1,:weight, -0.5*C_BG_Th)
add_edge!(g,2,5,:weight, C_Cor_BG_Th)
add_edge!(g,3,1,:weight, -0.5*C_BG_Th)
add_edge!(g,3,2,:weight, C_BG_Th)
add_edge!(g,4,3,:weight, -0.5*C_BG_Th)
add_edge!(g,4,4,:weight, C_BG_Th_Cor)
```
Now we are ready to build the ModelingToolkit System and apply structural simplification to the equations.

```@example Jansen-Rit
@named final_system = system_from_graph(g)
final_system_sys = structural_simplify(final_system)
```
Our Jansen-Rit model allows delayed edges, and we therefore need to collect those delays (in our case all delays are zero).  Then we build a Delayed Differential Equations Problem (DDEProblem).
```@example Jansen-Rit
sim_dur = 1000.0 # Simulate for 1 second
prob = ODEProblem(final_system_sys,
    [],
    (0.0, sim_dur))
```
We select an algorithm and solve the system
```@example Jansen-Rit
alg = Tsit5()
sol_dde_no_delays = solve(prob, alg, saveat=1)
plot(sol_dde_no_delays)
```
In a later tutorial, we will show how to introduce edge delays.
