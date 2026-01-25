# [Reinforcement Learning](@id api_rl)
```@meta
CurrentModule = Neuroblox
```

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
increment_trial!
```
