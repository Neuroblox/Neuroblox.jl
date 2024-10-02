# # Getting Started with Neuroblox

# This tutorial will introduce you to simulating brain dynamics using Neuroblox.
# In this example, we'll create a simple oscillating circuit using two Wilson-Cowan neural mass models [1]. The Wilson-Cowan model is one of the most influential models in computational neuroscience [2], describing the dynamics of interactions between populations of excitatory and inhibitory neurons.

# ## The Wilson-Cowan Model

# Each Wilson-Cowan neural mass is described by the following equations:

# ```math
# \begin{align}
# \nonumber
# \frac{dE}{dt} &= \frac{-E}{\tau_E} + S_E(c_{EE}E - c_{IE}I + \eta\textstyle\sum{jcn})\\[10pt]
# \nonumber
# \frac{dI}{dt} &= \frac{-I}{\tau_I} + S_I\left(c_{EI}E - c_{II}I\right)
# \end{align}
# ```

# where $E$ and $I$ denote the activity levels of the excitatory and inhibitory populations, respectively. The terms $\frac{dE}{dt}$ and $\frac{dI}{dt}$ describe the rate of change of these activity levels over time. The parameters $\tau_E$ and $\tau_I$ are time constants analogous to membrane time constants in single neuron models, determining how quickly the excitatory and inhibitory populations respond to changes. The coefficients $c_{EE}$ and $c_{II}$ represent self-interaction (or feedback) within excitatory and inhibitory populations, while $c_{IE}$ and $c_{EI}$ represent the cross-interactions between the two populations. The term $\eta\sum{jcn}$ represents external input to the excitatory population from other brain regions or external stimuli, with $\eta$ acting as a scaling factor. While $S_E$ and $S_I$ are sigmoid functions that represent the responses of neuronal populations to input stimuli, defined as:

# ```math
# S_k(x) = \frac{1}{1 + exp(-a_kx - \theta_k)}
# ```

# where $a_k$ and $\theta_k$ determine the steepness and threshold of the response, respectively.

# ## Building the Circuit

# Let's create an oscillating circuit by connecting two Wilson-Cowan neural masses:

using Neuroblox
using DifferentialEquations
using CairoMakie

## Create two Wilson-Cowan blox
@named WC1 = WilsonCowan()
@named WC2 = WilsonCowan()

## Create a graph to represent our circuit
g = MetaDiGraph()
add_blox!.(Ref(g), [WC1, WC2])

## Define the connectivity between the neural masses
add_edge!(g, WC1 => WC1; weight = -1) ## recurrent connection from WC1 to itself
add_edge!(g, WC1 => WC2; weight = 7) ## connection from WC1 to WC2
add_edge!(g, WC2 => WC1; weight = 4) ## connection from WC2 to WC1
add_edge!(g, WC2 => WC2; weight = -1) ## recurrent connection from WC2 to itself

# Here, we've created two Wilson-Cowan Blox and connected them as nodes in a directed graph. The `adj` matrix defines the weighted edges between these nodes. Each entry `adj[i,j]` represents how the output of blox `j` influences the input of blox `i`:
# 
# - Diagonal elements (`adj[1,1]` and `adj[2,2]`): Self-connections, adding feedback to each blox.
# - Off-diagonal elements (`adj[1,2]` and `adj[2,1]`): Inter-blox connections, determining how each blox influences the other.

# By default, the output of each Wilson-Cowan blox is its excitatory activity (E). The negative self-connections (-1) provide inhibitory feedback, while the positive inter-blox connections (6) provide strong excitatory coupling. This setup creates an oscillatory dynamic between the two Wilson-Cowan units.

# ## Creating the Model

# Now, let's build the complete model:

@named sys = system_from_graph(g)

# This creates a differential equations system from our graph representation using ModelingToolkit and symbolically simplifies it for efficient computation.

# ## Simulating the Model

# We are now ready to simulate our model. The following code creates and solves an `ODEProblem` for our system, simulating 100 time units of activity. In Neuroblox, the default time unit is milliseconds. We use `Rodas4`, a solver efficient for stiff problems. The solution is saved every 0.1 ms, allowing us to observe the detailed evolution of the system's behavior.

prob = ODEProblem(sys, [], (0.0, 100), [])
sol = solve(prob, Rodas4(), saveat=0.1)

# ## Plotting simulation results

# Finally, let us plot the `E` and `I` states of the first component, `WC1`. 

E1 = state_timeseries(WC1, sol, "E")
I1 = state_timeseries(WC1, sol, "I")

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time (ms)")
lines!(ax, sol.t, E1, label = "E")
lines!(ax, sol.t, I1, label = "I")
Legend(fig[1,2], ax)
fig

# [[1] Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical journal, 12(1), 1-24.](https://www.sciencedirect.com/science/article/pii/S0006349572860685)

# [[2] Destexhe, A., & Sejnowski, T. J. (2009). The Wilson–Cowan model, 36 years later. Biological cybernetics, 101(1), 1-2.](https://link.springer.com/article/10.1007/s00422-009-0328-3)
