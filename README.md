| **Documentation** | **Build Status** | **Citation** |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14192732.svg)](https://doi.org/10.5281/zenodo.14192732)

## Neuroblox.jl
## About
Neuroblox.jl is designed for computational neuroscience and psychiatry applications. Our tools range from control circuit system identification to brain circuit simulations bridging scales from spiking neurons to fMRI-derived circuits, parameter-fitting models to neuroimaging data, interactions between the brain and other physiological systems, experimental optimization, and scientific machine learning.

## Description
Neuroblox.jl is based on a library of modular computational building blocks (“blox”) in the form of systems of symbolic dynamic differential equations that can be combined to describe large-scale brain dynamics.  Once a model is built, it can be simulated efficiently and fit electrophysiological and neuroimaging data.  Moreover, the circuit behavior of multiple model variants can be investigated to aid in distinguishing between competing hypotheses.
We employ ModelingToolkit.jl to describe the dynamical behavior of blox as symbolic (stochastic/delay) differential equations.  Our libraries of modular blox consist of individual neurons (Hodgkin-Huxley, IF, QIF, LIF, etc.), neural mass models (Jansen-Rit, Wilson-Cowan, Lauter-Breakspear, Next Generation, microcanonical circuits etc.) and biomimetically-constrained control circuit elements.  A GUI designed to be intuitive to neuroscientists allows researchers to build models that automatically generate high-performance systems of numerical ordinary/stochastic differential equations from which one can run stimulations with parameters fit to experimental data.  Our benchmarks show that the increase in speed for simulation often exceeds a factor of 100 as compared to neural mass model implementation by the Virtual Brain (python** and similar packages in MATLAB.  For parameter fitting of brain circuit dynamical models, we use Turing.jl to perform probabilistic modeling, including Hamilton-Monte-Carlo sampling and Automated Differentiation Variational Inference.

## Installation

Neuroblox is available in the [NeurobloxRegistry](add https://github.com/Neuroblox/NeurobloxRegistry). In order to install the NeurobloxRegistry simply run

``` julia
using Pkg
pkg"registry add https://github.com/Neuroblox/NeurobloxRegistry"
```

and then Neuroblox can be installed like any other julia package with

``` julia
Pkg.add("Neuroblox)
```

If this is your first time using Julia, you *may* also need to add the General registry, which can be done with
```
pkg"registry add General"
```


## Licensing

Neuroblox is free for non-commerical and academic use. For full details of the license, please see 
[the Neuroblox EULA](https://github.com/Neuroblox/NeurobloxEULA). For commercial use, get in contact
with sales@neuroblox.org.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml
