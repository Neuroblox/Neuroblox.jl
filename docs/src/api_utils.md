# [Utility functions](@id api_utils) 
```@meta
CurrentModule = Neuroblox
```

Blox are the basic components of neural circuits. For documentation of the kinds of blox available, please see the [Blox documentation page](@ref api_blox). Internally these are represented as ModelingToolkit systems.

The following functions are used to query blox.
```@docs
get_neurons
get_exci_neurons
get_inh_neurons
nameof
namespaced_nameof
full_namespaced_nameof
```

Additionally there are several helpers for extracting useful information from solutions to simulations, such as the timing of spikes and the firing rate. Several of these are called by the plotting functions.

```@docs
detect_spikes
firing_rate
inter_spike_intervals
flat_inter_spike_intervals
powerspectrum
voltage_timeseries
meanfield_timeseries
state_timeseries
```
