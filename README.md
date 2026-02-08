# Neuroblox.jl

| **Documentation** | **Build Status** | **Citation** |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14192732.svg)](https://doi.org/10.5281/zenodo.14192732)

## About

[Neuroblox](neuroblox.ai) is a neuroCAD platform designed to accelerate biotech innovation and precision therapeutics for neuropsychiatric disorders and cognitive performance.  Optimize and derisk drugs, devices, and procedures in silico &mdash; before launching in patients.	

Start by loading prebuilt (experimentally validated) neural circuits that you can modify, or use experimental data to model your own neural circuits from modular components.  Next, simulate treatment effects across spatio-temporal scales and predict outcomes: from molecular targets to biomarkers to psychiatric/cognitive symptoms.  Neuroblox uses libraries of modular neurobiological building blocks ("blox") that snap together like components in a circuit diagram.

### What Neuroblox can simulate:
- Individual neurons: Hodgkin-Huxley, IF, QIF, LIF
- Neuroreceptor dynamics: Glu AMPA, GABA-A, GABA-B, NMDA, D1, D2, 5-HT, TRPM4, nicotinic ACh ($\alpha$, $\beta$), muscarinic ACh
- Neural mass models: Jansen-Rit, Wilson-Cowan, Next-Generation
- Multiscale biomimetic neural circuits composed of neurons, synapses, micro-assemblies, and neural mass models

### What you can do with Neuroblox:
- Build circuits through an intuitive GUI or Neuroblox's neuroscience-specific programming language 
- Simulate interventions (drugs, devices, sensory inputs, behavioral tasks) and track neurobehavioral effects across scales
- Fit model parameters to your experimental data (electrophysiology, neuroimaging, behavioral)
- Rigorously compare competing hypotheses with parallel testing of alternative circuit-architectures and mechanisms

Neuroblox models compile to high-performance numerical kernels, enabling practical exploration of large parameter spaces and running optimization experiments that would be infeasible on other platforms.

## Licensing

Neuroblox is free for non-commercial and academic use. For full details of the license, see 
the [Neuroblox EULA](https://github.com/Neuroblox/NeurobloxEULA). For commercial use, please contact
info@neuroblox.ai.

## Installation

Neuroblox is available in the [NeurobloxRegistry](https://github.com/Neuroblox/NeurobloxRegistry). In order to install the NeurobloxRegistry simply run

``` julia
using Pkg
pkg"registry add General"
pkg"registry add https://github.com/Neuroblox/NeurobloxRegistry"
```

and then Neuroblox can be installed like any other Julia package with

``` julia
Pkg.add("Neuroblox")
```

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml

