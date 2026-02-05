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

[Neuroblox](neuroblox.ai) is a computational platform for designing and testing brain interventions before they reach the clinic. Build neural circuit models from modular components, simulate treatment effects across biological scales, and predict outcomes from molecular targets to clinical symptoms.

Neuroblox is built on a library of modular computational building blocks ("blox") that snap together like components in a circuit diagram.

### What Neuroblox can model:
- Individual neurons (Hodgkin-Huxley, IF, QIF, LIF, etc.)
- Receptor dynamics (NMDA, Glutamate, GABA A/B, Dopamine, etc.)
- Neural mass models (Jansen-Rit, Wilson-Cowan, Next Generation models, etc.)
- Multi-scale, biomimetically-constrained neural circuits comprised of neurons, synapses, neural mass models, and other sub-circuits

### What you can do with Neuroblox:
- Build circuits through an intuitive GUI or programmatically
- Simulate interventions (drugs, devices, stimulation) and observe downstream effects
- Fit model parameters to your experimental data (electrophysiology, neuroimaging)
- Compare competing hypotheses by testing alternative circuit architectures

Neuroblox models compile to high-performance numerical kernels making it practical to explore large parameter spaces and run optimization experiments that would be infeasible on other platforms.

## Licensing

Neuroblox is free for non-commerical and academic use. For full details of the license, please see 
[the Neuroblox EULA](https://github.com/Neuroblox/NeurobloxEULA). For commercial use, get in contact
with info@neuroblox.ai.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml
