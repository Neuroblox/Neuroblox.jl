# [Neuroblox API](@id api_reference)
```@meta
CurrentModule = Neuroblox
```

This page documents the API functions available to users of Neuroblox. This page is focused on utilities like plotting, system generation and querying, and . For documentation of the various Blox, see the [Blox documentation page](@ref api_blox).

## [Model Creation and Querying](@id api_system)
At the highest level, a Neuroblox model consists of a weighted directed graph (represented as a MetaDiGraph)
whose nodes are Blox representing neurons and populations of neurons, and whose edges are connectors dictating 
how the dynamics of the Blox affect each other. The weights of the edges represent the strengths of synaptic 
connections. The model graph is used to generate a system of ordinary differential equations that can be solved 
using Julia's differential equations solvers.


### Graphs and Systems
The following functions are used in the construction of graphs.
```@docs
add_blox!
create_adjacency_edges!
```

The following functions are used in the construction of systems from graphs or lists of Blox.
```@docs
system_from_graph
system_from_parts
```

The following functions are used to query graphs.
```@docs
get_system
connectors_from_graph
generate_discrete_callbacks
```

### Blox
Blox are the basic components of neural circuits. For documentation of the kinds of blox available, please see the [Blox documentation page](@ref api_blox). Internally these are represented as ModelingToolkit systems.

The following functions are used to query blox.
```@docs
get_parts
get_neurons
get_exci_neurons
get_inh_neurons
get_system
get_input_equations
nameof
namespaced_nameof
inputs
outputs
equations
unknowns
parameters
discrete_events
```

### Connectors
Connectors connect between Blox. They are characterized by connection equations dictating how the state of one Blox affects the other, affects that are triggered when the source spikes, a weight representing the strength of the synaptic connection, and a learning rule dictating how that weight changes when the source fires.

When constructing the graph, the desired connections are represented as edges and added using `add_edge!`, but the Connectors are not instantiated until the full system is created using `system_from_graph`.

The following are used to query properties of connectors.
```@docs
equations
discrete_callbacks
sources
destinations
weights
delays
spike_affects
learning_rules
```

The following functions must be defined every time the user wants to define a new kind of connection between two types of Blox.
```@docs
connection_equations
connection_spike_affects
connection_learning_rule
connection_callbacks
```

Connection rules are used to define the structure of connections between composite blox (i.e. ones consisting of multiple neurons). They are passed in as a `connection_rule` keyword argument to `add_edge!`, which can be `:hypergeometric`, `:density`, or `:weightmatrix`. Internally, these keyword arguments correspond to the following functions. 
```@docs
hypergeometric_connections
density_connections
weight_matrix_connections
indegree_constrained_connections
```

## [Plotting](@id api_plotting)
This section documents helpers for generating plots from solutions to Neuroblox simulations. The backend for generating plots
for Neuroblox is [Makie](https://docs.makie.org/stable/). In order to call these functions, one must have a Makie backend installed, such
as CairoMakie or GLMakie.

```@docs
meanfield
meanfield!
rasterplot
rasterplot!
stackplot
stackplot!
frplot
frplot!
voltage_stack
powerspectrumplot
powerspectrumplot!
```

Additionally there are several helpers for extracting useful information from solutions to simulations, such as the timing of spikes and the firing rate. Several of these are called by the plotting functions.

```@docs
detect_spikes
mean_firing_rate
firing_rate
inter_spike_intervals
flat_inter_spike_intervals
powerspectrum
voltage_timeseries
meanfield_timeseries
state_timeseries
```

## [Reinforcement Learning](@id api_rl)
The following section documents the infrastructure for performing reinforcement learning on neural systems.
The neural system acts as the agent, while the environment is a series of sensory stimuli presented to the model.
The agent's action is the classification of the stimuli, and the choice follows some policy. Learning occurs as 
the connection weights of the system are updated according to some learning rule after each choice made by the 
agent.

```@docs
Agent
ClassificationEnvironment
```

The following policies are implemented in Neuroblox. Policies are represented by the `AbstractActionSelection` type.
Policies are added as nodes to the graph, with the set of actions represented by incoming connections.
```@docs
GreedyPolicy
action_selection_from_graph
```

The following learning rules are implemented in Neuroblox:
```@docs
HebbianPlasticity
HebbianModulationPlasticity
weight_gradient
```

The following functions are used to run reinforcement learning experiments.
```@docs
run_warmup
run_trial!
run_experiment!
reset!
increment_trial!
```
