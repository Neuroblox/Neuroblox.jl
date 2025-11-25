
# v0.7.0

## API Changes

+ Neuroblox.jl has been re-organized into a collection of sub-packages which can be used individually. This shouldn't cause any changes for the end user if they simply do `using Neuroblox` since Neuroblox itself now just re-exports the sub-packages. Currently our public subpackages are:
  + NeurobloxBase.jl fundamental shared infrastructure for making Neuroblox work
  + NeurobloxBasics.jl some common, generic neuroscience primitives
  + NeurobloxDBS.jl infrastructure and models specific to our work on deep brain stimulation
  + NeurobloxPharma.jl infeastructure and models specific to our pharmacological work and the corticostriatal circuit
  
+ The `run_experiment!` function now no longer takes a `verbose::Bool` keyword argument, but instead can take a `monitor` keyword argument, which is passed to the `update_monitor!` callback will be run after each experiment trial with information about that trial. We provide `ProgressMeterMonitor` as a simple replacement for the functionality that the old `verbose=true` option gave.

+ The naming of many objects from Neuroblox.jl has been updated and made more consistent. Abstract types now all start with `Abstract` in their type name, and we have removed the suffix `Blox` from the types that had them e.g. `HHNeuronExciBlox` was renamed to `HHNeuronExci`, and `NeuralMassBlox` was renamed `AbstractNeuralMass`.

+ A `@graph` macro has been added to simplify the process of making Neuroblox.jl graphs to pass to `system_from_graph`.
