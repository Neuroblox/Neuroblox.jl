# # Building a model of the Basal Ganglia using Neural Mass models

# In this example, we will construct a model of Parkinson's disease using Jansen-Rit neural mass models, based on the work of Liu et al. (2020) [1].
# More generally, this tutorial shows how to build a simple version of the cortico-basal ganglia-thalamocortical neural loop, including both the [direct and indirect pathways](https://www.ncbi.nlm.nih.gov/books/NBK537141/).
# In this tutorial, we will specifically cover:
# - How to create a Jansen-Rit neural mass model in Neuroblox
# - How to create blocks with different default sets of parameters
# - Using symbolic hyperparameters during graph definition
# - Visualizing neural mass outputs in both temporal and spectral plots 

# ## The Jansen-Rit Neural Mass Model

# The Jansen-Rit model [2] is another popular neural mass model that, like the [Wilson-Cowan model from the Getting Started](@ref getting_started_julia), describes the average activity of neural populations. 
# [This resource](https://link.springer.com/referenceworkentry/10.1007/978-1-4614-7320-6_65-2) is a good general introduction to the Jansen-Rit model if you'd like to read more.
# Each Jansen-Rit unit is defined by the following differential equations:

# ```math
# \begin{align*}
# \frac{dx}{dt} &= y-\frac{2}{\tau}x \\[10pt]
# \frac{dy}{dt} &= -\frac{x}{\tau^2} + \frac{H}{\tau} \left[2\lambda S(\textstyle\sum{jcn}) - \lambda\right]
# \end{align*}
# ```

# where $x$ represents the average postsynaptic membrane potential of the neural population, $y$ is an auxiliary variable, $\tau$ is the membrane time constant, $H$ is the maximum postsynaptic potential amplitude, $\lambda$ determines the maximum firing rate, and $\sum{jcn}$ represents the sum of all synaptic inputs to the population. The sigmoid function $S(x)$ models the population's firing rate response to input and is defined as:
# ```math
# S(x) = \frac{1}{1 + \text{exp}(-rx)}
# ```

# where $r$ controls the steepness of the sigmoid, affecting the population's sensitivity to input.

# This model consists of four main components:
# - A cortical column consisting of pyramidal cells ``PY``, an excitatory interneuron population ``EI``, and an inhibitory interneuron population ``II``
# - A striatal network consisting of two populations of medium spiny neurons, ``D1`` and ``D2`` (named for the different dopamine receptors), and fast-spiking interneurons ``FSI``
# - Subcortical structures including the subthalamic nucleus ``STH``, the external segment of the globus pallidus ``GPE``, and the internal segment of the globus pallidus ``GPI``
# - A thalamic oscillator ``Th``

# The connections between these oscillators are shown in the Neuroblox GUI below:
# NEED FIGURE HERE!!


# ## Setting Up the Model

# We're now going to set up this model using code. Let's start by importing the necessary libraries and defining our neural masses:

using Neuroblox
using OrdinaryDiffEq
using CairoMakie

# The original paper uses parameters in seconds, but all models in Neuroblox have milliseconds as their time unit. We therefore specify a scaling factor to use the paramters from the paper:

τ_factor = 1000; ## Convert time units from seconds to milliseconds

# Now we'll setup the neural masses. The values are all taken from the original paper [1], Table 1.
# Notice that some of these neural masses have the default Neuroblox parameters (e.g., ``PY`` uses default cortical parameters, so the user can simply specify ``cortical=true`` to access these).
# If you want to see the full list of defaults for the Jansen-Rit model, you can type ``?Jansen-Rit`` in your Julia REPL to view the docstring.

## Create the cortical oscillators
@named PY  = JansenRit(cortical=true)  ## default parameters cortical Jansen Rit blox
@named EI  = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=5, r=5)
@named II  = JansenRit(τ=2.0*τ_factor, H=60/τ_factor, λ=5, r=5)

## Create the striatal oscillators
@named D1 = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named D2 = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)
@named FSI = JansenRit(τ=0.0022*τ_factor, H=20/τ_factor, λ=300, r=0.3)

## Create the remaining subcortical oscillators
@named STH = JansenRit(τ=0.01*τ_factor, H=20/τ_factor, λ=500, r=0.1)
@named GPE = JansenRit(cortical=false) ## default parameters subcortical Jansen Rit blox
@named GPI = JansenRit(cortical=false)  ## default parameters subcortical Jansen Rit blox
@named Th  = JansenRit(τ=0.002*τ_factor, H=10/τ_factor, λ=20, r=5);

# ## Building the Circuit

# Now, let's create a graph representing our brain circuit. The nodes on this graph are the neural mass models defined aboe and the edges are the connections between the nodes based on the known anatomy of the basal ganglia-thalamocortical circuit.
# We define the connections specified in Table 2 of Liu et al. [1], although we only implement a subset of the nodes and edges to describe a simplified version of the model.
# Our connections share some common parameters which we define here, both as symbols and values, and use them as expressions for the weight of each connection :

g = MetaDiGraph() ## define an empty graph

params = @parameters C_Cor=60 C_BGTh=60 C_Cor➡BGTh=5 C_BGTh➡Cor=5 # define common connection parameters
## additional paramters for healthy version
##params = @parameters C_Cor=3 C_BGTh=3 C_Cor➡BGTh=9.75 C_BGTh➡Cor=9.75 # define common connection parameters

## Create connections
# Values come from Table 2, signs come from Figure 1

## thalamocortical connection
add_edge!(g, Th => EI; weight = C_BGTh➡Cor)

## remaining cortical → subcortical connections
add_edge!(g, PY => STH; weight = C_Cor➡BGTh)
add_edge!(g, PY => D1; weight = C_BGTh➡Cor)
add_edge!(g, PY => D2; weight = C_BGTh➡Cor)
add_edge!(g, PY => FSI; weight = C_BGTh➡Cor)

## basal ganglia ↔ thalamus connections
add_edge!(g, STH => GPE; weight = C_BGTh)
add_edge!(g, STH => GPI; weight = C_BGTh)
add_edge!(g, GPE => STH; weight = -0.5*C_BGTh)
add_edge!(g, GPE => GPE; weight = -0.5*C_BGTh)
add_edge!(g, GPE => GPI; weight = -0.5*C_BGTh)
add_edge!(g, GPE => FSI; weight = -0.5*C_BGTh)
add_edge!(g, FSI => D1; weight = -0.5*C_BGTh)
add_edge!(g, FSI => D2; weight = -0.5*C_BGTh)
add_edge!(g, FSI => FSI; weight = -0.5*C_BGTh)
add_edge!(g, D1 => D1; weight = -0.5*C_BGTh)
add_edge!(g, D1 => D2; weight = -0.5*C_BGTh)
add_edge!(g, D1 => GPI; weight = -0.5*C_BGTh)
add_edge!(g, D2 => D2; weight = -0.5*C_BGTh)
add_edge!(g, D2 => D1; weight = -0.5*C_BGTh)
add_edge!(g, D2 => GPE; weight = -0.5*C_BGTh)
add_edge!(g, GPI => Th; weight = -0.5*C_BGTh)

## corticocortical connections
add_edge!(g, PY => EI; weight = 6*C_Cor)
add_edge!(g, PY => II; weight = 1.5*C_Cor)
add_edge!(g, EI => PY; weight = 4.8*C_Cor)
add_edge!(g, II => PY; weight = -1.5*C_Cor)
add_edge!(g, II => II; weight = -3.3*C_Cor)

# ## Creating the Model

# Let's build the complete model:

@named final_system = system_from_graph(g)

# This creates a differential equations system from our graph representation using ModelingToolkit and symbolically simplifies it for efficient computation.

# ## Simulating the Model

# Lastly, we create the `ODEProblem` for our system, select an algorithm, in this case `Tsit5()`, and simulate 1 second of brain activity.

sim_dur = 1000.0 # Simulate for 1 second
prob = ODEProblem(final_system, [], (0.0, sim_dur))
sol = solve(prob, Tsit5(), saveat=1)


# ## References
# [1] Liu, C, Zhou, C, Wang, J, Fietkiewicz, C, & Loparo, KA. (2020). The role of coupling connections in a model of the cortico-basal ganglia-thalamocortical neural loop for the generation of beta oscillations. Neural Networks, 123, 381-392. DOI: [10.1016/j.neunet.2019.12.021](https://doi.org/10.1016/j.neunet.2019.12.021)
# [2] Jansen BH, Rit VG. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biol Cybern. 1995 Sep;73(4):357-66. DOI: [10.1007/BF00199471](https://pubmed.ncbi.nlm.nih.gov/7578475/).