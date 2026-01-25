# [Model Graphs](@id api_reference)
```@meta
CurrentModule = Neuroblox
```

At the highest level, a Neuroblox model consists of a weighted directed graph (represented as a MetaDiGraph)
whose nodes are Blox representing neurons and populations of neurons, and whose edges are connectors dictating 
how the dynamics of the Blox affect each other. The weights of the edges represent the strengths of synaptic 
connections. The model graph is used to generate a system of ordinary differential equations that can be solved 
using Julia's differential equations solvers.

Users can also save/load a graph model and/or an ODE problem that contains the model in addition to simulation parameters like a timespan, initial conditions, etc.

## Building a model from a graph
```@docs
@graph
@graph!
```

## Saving/loading a model
```@docs
save_graph
load_graph
@save_graph
save_problem
load_problem
```
