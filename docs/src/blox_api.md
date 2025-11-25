# [Blox Documentation](@id api_blox)
```@meta
CurrentModule = Neuroblox
```

This page is a list of all of the Blox that Neuroblox can create.

A blox can be a neuron, a neural mass model, a receptor, a composite circuit of the preceding, a stimulus or current input, or experimental readouts (observers).

## [Neuron Blox](@id api_neuron_blox)
The following section documents some of the basic types of neurons that can be created, along with a description and parameters that can be set.

```@docs
IFNeuron
LIFNeuron
QIFNeuron
IzhikevichNeuron
LIFExciNeuron
LIFInhNeuron
PINGNeuronExci
PINGNeuronInhib
HHNeuronExci
HHNeuronInhib
HHNeuronFSI
MetabolicHHNeuron
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

## [Composite Blox](@id api_composite_blox)
Composite blox typically represent larger-scale brain structures.

### Pharmaceutical Module
The following composite blox are used for simulations of the effects of drug dosing on neural circuits.

```@docs
WinnerTakeAll
Cortical
LateralAmygdalaCluster
LateralAmygdala
CentralAmygdala
Striatum
GPi
GPe
Thalamus
STN
Nuc_Reticularis
```

### Deep Brain Stimulation
The following composite blox are used for simulations of deep brain stimulation.

```@docs
HHNeuronInhib_MSN_Adam
HHNeuronInhib_FSI_Adam
HHNeuronExci_STN_Adam
HHNeuronInhib_GPe_Adam
Striatum_MSN_Adam
Striatum_FSI_Adam
GPe_Adam
STN_Adam
```

## [Discrete Blox](@id api_discrete_blox)
Discrete blox are ones in which the state changes discretely, rather than according to a differential equation.

```@docs
Matrisome
Striosome
TAN
SNc
```

## [Receptor Blox](@id api_receptor_blox)

```@docs
MoradiNMDAR
```

## [Source Blox](@id api_source_blox)
Sources are external inputs that feed into neural circuits. These may represent electrical inputs or sensory stimuli.


### External Current Inputs
```@docs
ConstantInput
PoissonSpikeTrain
VoltageClampSource
ARProcess
PulsesInput
```

### Deep Brain Stimulation Stimuli
```@docs
DBS
ProtocolDBS
SquareStimulus
BurstStimulus
```

### Sensory Stimuli
```@docs
ImageStimulus
```

## Observers
Observers are experimental readouts from neural simulations, such as fMRI signals.

```@docs
BalloonModel
boldsignal_endo_balloon
```
