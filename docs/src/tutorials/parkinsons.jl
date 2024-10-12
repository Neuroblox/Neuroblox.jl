# # Building a model of Parkinson's disease using Neural Mass models

# In this example, we'll construct a model of Parkinson's disease using eight Jansen-Rit Neural Mass Models, based on the work of Liu et al. (2020) [1].

# ## The Jansen-Rit Neural Mass Model

# The Jansen-Rit model [2] is another popular neural mass model that, like the Wilson-Cowan model from [Example 1](#example-1-creating-a-simple-neural-circuit), describes the average activity of neural populations. Each Jansen-Rit unit is defined by the following differential equations:

# ```math
# \begin{align}
# \frac{dx}{dt} &= y-\frac{2}{\tau}x \\[10pt]
# \frac{dy}{dt} &= -\frac{x}{\tau^2} + \frac{H}{\tau} \left[2\lambda S(\textstyle\sum{jcn}) - \lambda\right]
# \end{align}
# ```

# where $x$ represents the average postsynaptic membrane potential of the neural population, $y$ is an auxiliary variable, $\tau$ is the membrane time constant, $H$ is the maximum postsynaptic potential amplitude, $\lambda$ determines the maximum firing rate, and $\sum{jcn}$ represents the sum of all synaptic inputs to the population. The sigmoid function $S(x)$ models the population's firing rate response to input and is defined as:


# ```math
# S(x) = \frac{1}{1 + \text{exp}(-rx)}
# ```

# where $r$ controls the steepness of the sigmoid, affecting the population's sensitivity to input.

# ## Setting Up the Model

# Let's start by importing the necessary libraries and defining our neural masses:

using Neuroblox
using OrdinaryDiffEq
using CairoMakie

## Convert time units from seconds to milliseconds
τ_factor = 1000

## Define Jansen-Rit neural masses for different brain regions
@named Str = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named GPE = JansenRit(τ=0.04*τ_factor, cortical=false)  # all default subcortical except τ
@named STN = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1)
@named GPI = JansenRit(cortical=false)  # default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5)
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5)
@named PY  = JansenRit(cortical=true)  # default parameters cortical Jansen Rit blox
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5)

#Here, we've created eight Jansen-Rit neural masses representing different brain regions involved in Parkinson's disease. The `τ_factor` is used to convert time units from seconds (as in the original paper) to milliseconds (Neuroblox's default time unit).

# ## Building the Circuit

# Now, let's create a graph representing our brain circuit. The nodes on this graph are the neural mass models defined aboe and the edges are the connections between the nodes based on the known anatomy of the basal ganglia-thalamocortical circuit.
# As an alternative to creating edges with an adjacency matrix, here we demonstrate a different approach by adding edges one by one. In this case, we set the connections specified in Table 2 of Liu et al. [1], although we only implement a subset of the nodes and edges to describe a simplified version of the model.
# Our connections share some common parameters which we define here, both as symbols and values, and use them as expressions for the weight of each connection :

g = MetaDiGraph() ## define an empty graph

params = @parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5 # define common connection parameters

## Create connections
add_edge!(g, GPE => Str; weight = -0.5*C_BG_Th)
add_edge!(g, GPE => GPE; weight = -0.5*C_BG_Th)
add_edge!(g, GPE => STN; weight = C_BG_Th)
add_edge!(g, STN => GPE; weight = -0.5*C_BG_Th)
add_edge!(g, STN => PY; weight = C_Cor_BG_Th)
add_edge!(g, GPI => GPE; weight = -0.5*C_BG_Th)
add_edge!(g, GPI => STN; weight = C_BG_Th)
add_edge!(g, Th => GPI; weight = -0.5*C_BG_Th)
add_edge!(g, EI => Th; weight = C_BG_Th_Cor)
add_edge!(g, EI => PY; weight = 6*C_Cor)
add_edge!(g, PY => EI; weight = 4.8*C_Cor)
add_edge!(g, PY => II; weight = -1.5*C_Cor)
add_edge!(g, II => PY; weight = 1.5*C_Cor)
add_edge!(g, II => II; weight = 3.3*C_Cor)
add_edge!(g, Str => Str; weight = -0.5*C_BG_Th)
add_edge!(g, Str => GPE; weight = C_BG_Th)
add_edge!(g, GPE => Str; weight = -0.5*C_BG_Th)
add_edge!(g, GPE => Th; weight = C_Cor_BG_Th)
add_edge!(g, STN => Str; weight = -0.5*C_BG_Th)
add_edge!(g, STN => GPE; weight = C_BG_Th)
add_edge!(g, GPI => STN; weight = -0.5*C_BG_Th)
add_edge!(g, GPI => GPI; weight = C_BG_Th_Cor)

# ## Creating the Model

# Let's build the complete model:

@named final_system = system_from_graph(g)

# This creates a differential equations system from our graph representation using ModelingToolkit and symbolically simplifies it for efficient computation.

# ## Simulating the Model

# Lastly, we create the `ODEProblem` for our system, select an algorithm, in this case `Tsit5()`, and simulate 1 second of brain activity.

sim_dur = 1000.0 # Simulate for 1 second
prob = ODEProblem(final_system, [], (0.0, sim_dur))
sol = solve(prob, Tsit5(), saveat=1)


# [[1] Liu, C., Zhou, C., Wang, J., Fietkiewicz, C., & Loparo, K. A. (2020). The role of coupling connections in a model of the cortico-basal ganglia-thalamocortical neural loop for the generation of beta oscillations. Neural Networks, 123, 381-392.](https://doi.org/10.1016/j.neunet.2019.12.021)