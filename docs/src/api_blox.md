# [Neuron, Neural Mass and Composite Blox](@id api_blox)
```@meta
CurrentModule = Neuroblox
```

This page is a list of all of the Blox that Neuroblox can create.

A blox can be a neuron, a neural mass model, a receptor, a composite circuit of the preceding, a stimulus or current input, or experimental readouts (observers).

## [Neuron Blox](@id api_neuron_blox)
The following section documents the types of neurons that can be created, along with a description and parameters that can be set.

First there are the ion-channel based neuron types

```@docs
HHNeuronExci
HHNeuronInhib
BaxterSensoryNeuron
TRNNeuron
MuscarinicNeuron
VTADANeuron
VTAGABANeuron
PINGNeuronExci
PINGNeuronInhib
```

We also have a range of more phenomenological neuron models, with event-based spiking.

```@docs
IFNeuron
LIFNeuron
QIFNeuron
IzhikevichNeuron
LIFExciNeuron
LIFInhNeuron
```

## [Neural Mass Model Blox](@id api_nmm_blox)
Neural mass models simulate the average activity and voltage of a collection of neurons.

```@docs
NGNMM_Izh
NGNMM_QIF
LinearNeuralMass
HarmonicOscillator
JansenRit
WilsonCowan
LarterBreakspear
KuramotoOscillator
VanDerPol
Generic2dOscillator
OUProcess
NGNMM_theta
NextGenerationEI
```

## [Discrete Dynamics Blox](@id api_blox_discrete)

```@docs
Matrisome
Striosome
SNc
TAN
```

## [Composite Blox](@id api_composite_blox)
Composite blox typically represent larger-scale brain structures. They may contain multiple neurons, neural masses, sources, or other types of blox as their own internal circuits.

```@docs
WinnerTakeAll
Cortical
Striatum
GPi
GPe
Thalamus
STN
Nuc_Reticularis
```

## [Source Blox](@id api_source_blox)
Sources are external inputs that feed into neural circuits. These may represent electrical inputs or sensory stimuli.


### External Current Inputs
```@docs
ConstantInput
PoissonSpikeTrain
VoltageClampSource
PulsesInput
```

### Deep Brain Stimulation Stimuli
```@docs
DBS
ProtocolDBS
```

### Sensory Stimuli
```@docs
ImageStimulus
```

## Observers
Observers are experimental readouts from neural simulations, such as fMRI signals.

```@docs
BalloonModel
```
