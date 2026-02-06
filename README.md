# Neuroblox.jl

| **Documentation** | **Build Status** | **Citation** |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14192732.svg)](https://doi.org/10.5281/zenodo.14192732)

## Installation

Neuroblox is available in the [NeurobloxRegistry](https://github.com/Neuroblox/NeurobloxRegistry). In order to install the NeurobloxRegistry simply run

``` julia
using Pkg
pkg"registry add General"
pkg"registry add https://github.com/Neuroblox/NeurobloxRegistry"
```

and then Neuroblox can be installed like any other julia package with

``` julia
Pkg.add("Neuroblox")
```

## About

[Neuroblox](neuroblox.ai) is a computational platform for engineering and testing brain interventions before they reach the clinic. Integrate and operationalize data across scales, generate and test hypotheses, design and optimize novel neurotherapeutics, and inform precision medicine.  Build or load pre-built neural circuit models from modular components, simulate treatment effects across biological scales, and predict outcomes: from molecular targets to clinical symptoms (cognition, mood, and behavior).

Neuroblox is built on a library of modular computational building blocks ("blox") that snap together like components in a circuit diagram.

### What Neuroblox can model:
- Individual neurons (e.g., Hodgkin-Huxley, IF, QIF, LIF)
- Pharma and disorder-specific modulation via neuroreceptor dynamics (e.g., Glu AMPA, GABA A, GABA B, NMDA, D1, D2, 5-HT, TRPM4, nicotinic ACh (alpha, beta), muscarinic ACh)
- Neural mass models (e.g., Jansen-Rit, Wilson-Cowan, Next-Generation)
- Multiscale biomimetic neural circuits composed of neurons, synapses, micro-assemblies (including biomimetic computational primitives), and neural mass models.

### What you can do with Neuroblox:
- Build circuits through an intuitive GUI or Neuroblox's neuroscience-specific programming language 
- Simulate interventions (drugs, devices, sensory inputs, behavioral tasks) and observe neurobehavioral effects across scales
- Fit model parameters to your experimental data (electrophysiology, neuroimaging, behavioral)
- Rigorously compare competing hypotheses by parallel testing of alternative circuit-architectures and mechanisms

Neuroblox models compile to high-performance numerical kernels, enabling practical exploration of large parameter spaces and running optimization experiments that would be infeasible on other platforms.

## Licensing

Neuroblox is free for non-commercial and academic use. For full details of the license, please see 
[the Neuroblox EULA](https://github.com/Neuroblox/NeurobloxEULA). For commercial use, please contact
info@neuroblox.ai.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml
