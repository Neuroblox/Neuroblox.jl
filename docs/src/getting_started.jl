# # Getting Started

# ## Getting Started with Julia

# Here we would like to summarize some resources for people that are interested in learning more about the Julia language before or while exploring Neuroblox. Please follow the links below for introductory material on the language that is inclusive to all users; people familiar with programming or not, people with a mathematics, engineering, or science background :
# - [Introduction to Julia](https://youtu.be/7hVV5uoEo-0?si=JdMSCh3R4w2cl5uT) by Matt Bauman at the JuliaCon 2024
# - [Julia Tutorials & Workshops](https://julialang.org/learning/tutorials/), a collection of training materials from the official Julia website.
# - [Modern Julia Workflows](https://modernjuliaworkflows.org/), an introduction to how to write and share your Julia code effectively with tips & tricks.

# ## [Getting Started with Neuroblox](@id getting_started_julia)

# This example will introduce you to simulating brain dynamics using Neuroblox. We will create a simple oscillating circuit using two Wilson-Cowan neural mass models [1]. The Wilson-Cowan model is one of the most influential models in computational neuroscience [2], describing the dynamics of interactions between populations of excitatory and inhibitory neurons.

# ### The Wilson-Cowan Model

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

# ### Building the Circuit

# Let's create an oscillating circuit by connecting two Wilson-Cowan neural masses:

using Neuroblox
using OrdinaryDiffEqRosenbrock
using CairoMakie

## Create a graph of two Wilson-Cowan blox to represent our circuit
@graph g begin
    @nodes begin
        WC1 = WilsonCowan()
        WC2 = WilsonCowan()
    end
    @connections begin
        WC1 => WC1, [weight = -1] ## recurrent connection from WC1 to itself
        WC1 => WC2, [weight =  7] ## connection from WC1 to WC2
        WC2 => WC1, [weight =  4] ## connection from WC2 to WC1
        WC2 => WC2, [weight = -1] ## recurrent connection from WC2 to itself
    end
end
# Here, we've created two Wilson-Cowan blox and connected them as nodes in a directed graph. Each blox connects to itself and to the other blox. 

# By default, the output of each Wilson-Cowan blox is its excitatory activity (E). The negative self-connections (-1) provide inhibitory feedback, while the positive inter-blox connections (6) provide strong excitatory coupling. This setup creates an oscillatory dynamic between the two Wilson-Cowan units.

# ### Creating the Model

# Now, let's build the complete model:

prob = ODEProblem(g, [], (0.0, 100), [])

# This creates an efficient differential equations system from our graph representation using GraphDynamics.

# ### Simulating the Model

# We are now ready to simulate our model. The following code solvesthe `ODEProblem` for our system, simulating 100 time units of activity. In Neuroblox, the default time unit is milliseconds. We use `Rodas4`, a solver efficient for stiff problems. The solution is saved every 0.1 ms, allowing us to observe the detailed evolution of the system's behavior.

sol = solve(prob, Rodas4(), saveat=0.1)

# ### Plotting simulation results

# Finally, let us plot the `E` and `I` states of the first component, `WC1`. To do this we will use the `state_timeseries` function that extracts the timeseries of a blox state from the solution object. 

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
