# # Resting state simulation using neural mass models

# This tutorial will introduce you to simulating resting state brain dynamics using Neuroblox. We will be using the FitzHugh-Nagumo model as a building block. The FitzHugh-Nagumo model is described by the follwoing equations:

# ```math
#        \begin{align}
#        \dot{V} &= d \, \tau (-f V^3 + e V^2 + \alpha W - \gamma I_{c} + \sigma w(t) ) \\
#        \dot{W} &= \dfrac{d}{\tau}\,\,(b V - \beta W + a + \sigma w(t) )
#        \end{align}
# ```

# We start by building the resting state circuit from individual Generic2dOscillator Blox

using Neuroblox
using CSV
using DataFrames
using DifferentialEquations
using Random
using CairoMakie
using Statistics
using HypothesisTests

## read connection matrix from file
weights = CSV.read("../data/weights.csv",DataFrame)
region_names = names(weights)

wm = Array(weights) ## transform the weights into a matrix
N_bloxs = size(wm)[1] ## number of blox components

## create an array of neural mass models
blocks = [Generic2dOscillator(name=Symbol(region_names[i]),bn=sqrt(5e-4)) for i in 1:N_bloxs]

## add neural mass models to Graph and connect using the connection matrix
g = MetaDiGraph()
add_blox!.(Ref(g), blocks)
create_adjacency_edges!(g, wm)

@named sys = system_from_graph(g);

# To solve the system, we first create an Stochastic Differential Equation Problem and then solve it using a EulerHeun solver. The solution is saved every 0.5 ms. The unit of time in Neuroblox is 1 ms.

prob = SDEProblem(sys,rand(-2:0.1:4,76*2), (0.0, 6e5), [])
sol = solve(prob, EulerHeun(), dt=0.5, saveat=5)

# Let us plot the voltage potential of the first couple of components

v1 = voltage_timeseries(blocks[1], sol)
v2 = voltage_timeseries(blocks[2], sol)

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Potential")
lines!(ax, sol.t, v1)
lines!(ax, sol.t, v2)
xlims!(ax, (0, 1000)) ## limit the x-axis to the first second of simulation
fig

# To evaluate the connectivity of our simulated resting state network, we calculate the statistically significant correlations

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

heatmap(log10.(p) .* (p .< 0.05))
# Fig.: log10(p value) displaying statistally significant correlation between time series

heatmap(wm)
# Fig.: Connection Adjacency Matrix that was used to connect the neural mass models
