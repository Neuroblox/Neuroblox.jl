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
using StochasticDiffEq
using Random
using CairoMakie
using Statistics
using HypothesisTests
using Downloads
using SparseArrays

Random.seed!(123); # Set a fixed RNG seed for reproducible results

## download and read connection matrix from a file
weights = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/weights.csv"), DataFrame)
region_names = names(weights)

wm = Matrix(weights) ## transform the weights into a matrix
N_bloxs = size(wm)[1]; ## number of blox components

# You can visualize the sparsity structure by converting the weights matrix into a sparse matrix 
wm_sparse = SparseMatrixCSC(wm)

# After the connectivity structure, it's time to define the neural mass components of our model and then use the weight matrix to connect them together into our final system.

## create an array of neural mass models
blox = [Generic2dOscillator(name=Symbol(region_names[i]),bn=sqrt(5e-4)) for i in 1:N_bloxs]

## add neural mass models to Graph and connect using the connection matrix
g = MetaDiGraph()
add_blox!.(Ref(g), blox)
create_adjacency_edges!(g, wm)

@named sys = system_from_graph(g);

# To solve the system, we first create an Stochastic Differential Equation Problem and then solve it using a EulerHeun solver. The solution is saved every 0.5 ms. The unit of time in Neuroblox is 1 ms.

tspan = (0.0, 6e5)
prob = SDEProblem(sys,rand(-2:0.1:4,76*2), tspan, [])
sol = solve(prob, EulerHeun(), dt=0.5, saveat=5);

# Let us now plot the voltage potential of the first couple of components. We can extract the voltage timeseries of a blox from the solution object using the `voltage_timeseries` function.

v1 = voltage_timeseries(blox[1], sol)
v2 = voltage_timeseries(blox[2], sol)

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Potential")
lines!(ax, sol.t, v1)
lines!(ax, sol.t, v2)
xlims!(ax, (0, 1000)) ## limit the x-axis to the first second of simulation
fig

# To evaluate the connectivity of our simulated resting state network, we calculate the statistically significant correlations

step_sz = 1000
time_steps = range(1, length(sol.t); step = step_sz)

cs = Array{Float64}(undef, N_bloxs, N_bloxs, length(time_steps)-1)

## First we get the voltage timeseries of all blox in time bins 
for (i, t) in enumerate(time_steps[1:end-1])
    ## Get the voltage timeseries of all blox within a time window in a matrix 
    V = voltage_timeseries(blox, sol; ts = t:(t + step_sz))
    ## Calculate the correlation matrix of all timeseries per time window
    cs[:,:,i] = cor(V)
end

p = zeros(N_bloxs, N_bloxs)
for i in 1:N_bloxs
    for j in 1:N_bloxs
        ## Perform a t-test on the voltage correlations across all time windows
        p[i,j] = pvalue(OneSampleTTest(cs[i,j,:]))
    end
end

fig = Figure()
ax = [
    Axis(fig[1,1]; title = "Correlation p-values", aspect = AxisAspect(1)),
    Axis(fig[1,2]; title = "Adjacency matrix", aspect = AxisAspect(1))
]
## Use logical indexing to only plot the statistically significan p-values (p .< 0.05)
heatmap!(ax[1], log10.(p) .* (p .< 0.05))

heatmap!(ax[2], wm)
fig

# Notice how the correlation heatmap qualitatively matches the sparsity structure that we printed above with `wm_sparse`.
