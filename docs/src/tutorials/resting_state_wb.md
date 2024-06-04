# [Tutorial on resting state simulation using neural mass models](@id resting_state_tutorial)

This tutorial will introduce you to simulating resting state brain dynamics using Neuroblox.

## Building the a whole brain FitzHugh-Nagumo neural mass model

The FitzHugh-Nagumo model is described by the follwoing equations:

```math
        \begin{align}
        \dot{V} &= d \, \tau (-f V^3 + e V^2 + \alpha W - \gamma I_{c} + \sigma w(t) ) \\
        \dot{W} &= \dfrac{d}{\tau}\,\,(b V - \beta W + a + \sigma w(t) )
        \end{align}
```

We start by building the resting state circuit from individual Generic2dOscillator Blox

```@example resting-state-circuit
using Neuroblox
using CSV
using DataFrames
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics
using HypothesisTests

# read connection matrix from file
weights = CSV.read("../data/weights.csv",DataFrame)
region_names = names(weights)

wm = Array(weights)

# assemble list of neural mass models
blocks = []
for i in 1:size(wm)[1]
    push!(blocks, Neuroblox.Generic2dOscillator(name=Symbol(region_names[i]),bn=sqrt(5e-4)))
end

# add neural mass models to Graph and connect using the connection matrix
g = MetaDiGraph()
add_blox!.(Ref(g), blocks)
create_adjacency_edges!(g, wm)
```

```@example resting-state-circuit
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
length(unknowns(sys))
```

To solve the system, we first create an Stochastic Differential Equation Problem and then solve it over the tspan of (0,6e) using a EulerHeun solver.  The solution is saved every 0.5ms. The unit of time in Neuroblox is 1ms.

```@example resting-state-circuit
prob = SDEProblem(sys,rand(-2:0.1:4,76*2), (0.0, 6e5), [])
sol = solve(prob, EulerHeun(), dt=0.5, saveat=5)
plot(sol.t,sol[5,:],xlims=(0,10000))
```
To evaluate the connectivity of our simulated resting state network, we calculate the statistically significant correlations

```@example resting-state-circuit
cs = []
for i in 1:Int((length(sol.t)-1)/1000)-1
    solv = Array(sol[1:2:end,(i-1)*1000+1:(i*1000)])'
    push!(cs,cor(solv))
end
css = stack(cs)

p = zeros(76,76)
for i in 1:76
    for j in 1:76
        p[i,j] = pvalue(OneSampleTTest(css[i,j,:]))
    end
end
heatmap(log10.(p) .* (p .< 0.05),aspect_ratio = :equal)
```
Fig.: log10(p value) displaying statistally significant correlation between time series
```@example resting-state-circuit
heatmap(wm,aspect_ratio = :equal)
```
Fig.: Connection Adjacency Matrix that was used to connect the neural mass models
