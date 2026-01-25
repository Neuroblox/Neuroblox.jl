# v0.8.0

## API Changes

+ ModelingToolkit is no longer used by Neuroblox. All interfaces have moved over to GraphDynamics.jl as the backend.
+ A `@blox` macro has been added for defining blox types compatible with GraphDynamics
+ A `@wiring_rule` macro has been added for defining what should happen when two blox are connected
+ A `@connection` macro has been added for defining the action of connection types on subsystems, along with event connections
+ Instead of storing a system and connector object, composite blox (i.e. `<:AbstractComposite`) now store a `graph` field which holds a `GraphSystem` of the internal connections.
+ A `@graph!` macros have been added for modifying already existing `GraphSystem`s.
+ `Agent` and `ClassificationEnvironment` no longer take `name` and `namespace` keyword arguments.
+ An `@experiment` macro has been added to easily modify many parameters of an `ODEProblem`
+ Blox now support continuous events
+ Blox which used discrete events for spike detection, now set `dtmax` parameters which limit the maximum timestep a solver is allowed to take.
+ `GraphSystem`s and `ODEProblem`s can now be saved and loaded using `save_graph`, `load_graph`, `save_problem`, and `load_problem`
+ A `MultipointConnection` type has been added which allows one to write connection rules between two blox that reference other extra blox.
+ A large library of receptor types has been added to NeurobloxPharma including
  + `GABA_B_Synapse`
  + `NMDA_Synapse`
  + `MorandiFullNMDAR`
  + `MorandiNMDAR`
  + `MsnNMDAR`
  + `MsnAMPAR`
  + `MsnD1Receptor`
  + `MsnD2Receptor`
  + `HTR5`
  + `CaTRPM4R`
  + `Alpha7ERnAChR`
  + `MuscarinicR`
  + `Beta2nAChR`
+ Sevaral new neuron types have been added for interacting with the receptor library
  + `BaxterSensoryNeuron`
  + `TRNNeuron`
  + `MuscarinicNeuron`
  + `VTADANeuron`
  + `VTAGABANeuron`
+ The API pages in our documentation have been restructured and extended with a lot more docstrings. There are now new pages under API for :
 + Neuron, Neural mass and Composite blox definitions
 + Constructing, saving and loading graphs
 + Receptor definitions 
 + Reinforcement learning 
 + Plotting recipes
 + Utility functions
 + Adding user-defined blox
 + Adding user-defined connections
 + Defining experiments by changing model parameters and initial conditions
+ A tutorial on choosing receptors between neuronal connections was added in the documentation.

# v0.7.0

## API Changes

+ Neuroblox.jl has been re-organized into a collection of sub-packages which can be used individually. This shouldn't cause any changes for the end user if they simply do `using Neuroblox` since Neuroblox itself now just re-exports the sub-packages. Currently our public subpackages are:
  + NeurobloxBase.jl fundamental shared infrastructure for making Neuroblox work
  + NeurobloxBasics.jl some common, generic neuroscience primitives
  + NeurobloxDBS.jl infrastructure and models specific to our work on deep brain stimulation
  + NeurobloxPharma.jl infrastructure and models specific to our pharmacological work and the corticostriatal circuit
  
+ The `run_experiment!` function now no longer takes a `verbose::Bool` keyword argument, but instead can take a `monitor` keyword argument, which is passed to the `update_monitor!` callback will be run after each experiment trial with information about that trial. We provide `ProgressMeterMonitor` as a simple replacement for the functionality that the old `verbose=true` option gave.

+ The naming of many objects from Neuroblox.jl has been updated and made more consistent. Abstract types now all start with `Abstract` in their type name, and we have removed the suffix `Blox` from the types that had them e.g. `HHNeuronExciBlox` was renamed to `HHNeuronExci`, and `NeuralMassBlox` was renamed `AbstractNeuralMass`.

+ A `@graph` macro has been added to simplify the process of making Neuroblox.jl graphs to pass to `system_from_graph`.
