# Neuroblox 0.9 Beta Release

## A visual environment for building, simulating, and analyzing biophysically grounded brain circuits

NeurobloxGUI is the interactive modeling environment for the Neuroblox neuroCAD platform. The Neuroblox 0.9 release makes Neuroblox's circuit-modeling capabilities available through a visual, point-and-click interface, enabling researchers to harness the power of visual intuitions in constructing and simulating detailed brain circuits without the need for code. It is intended for computational and experimental neuroscience researchers, as well as pharmaceutical and biotechnology teams testing central nervous system (CNS) therapeutics *in silico*.

Circuits are assembled from dynamical "blox" — individual neurons, neural assemblies, and neural mass models — and simulated to predict how the underlying system behaves. This supports mechanism-of-action studies, target and biomarker exploration, and the *in silico* evaluation of candidate interventions across molecular, cellular, and circuit scales.

Installation instructions can be found [here](https://github.com/Neuroblox/Neuroblox.jl/blob/main/INSTALLATION.md)

https://github.com/user-attachments/assets/13a4c325-4f43-45ca-8bba-414ca52c961b

> [!NOTE]
> **Neuroblox 0.95 (1 August 2026)** will contain the Neuroblox Domain Specific Language (DSL): a new programming language designed to be highly intuitive and user-friendly for neuroscientists. In Neuroblox 0.95, the NeurobloxGUI will include an Integrated Development Environment (IDE) to permit users to interact with the GUI via the Neuroblox DSL. Neuroblox 0.95 users are highly encouraged to provide feedback by contacting <support@neuroblox.ai>; this feedback will be incorporated into **Neuroblox 1.0 (1 October 2026)**.

## Guided tutorials and worked examples

The release includes a comprehensive guided tutorial that introduces the environment's core functions and then reproduces most of the built-in circuit examples step by step. These include experimentally validated models such as the corticostriatal micro-assembly circuit and the propofol anesthesia example. This allows new users to become proficient using neurotherapeutically relevant circuits capable of generating clinically-relevant outputs at the human biomarker (neuroimaging) and behavioral scales.

## Circuit design on the flatbrain canvas

Models are built on the **flatbrain canvas**, a Mercator-style projection of the brain that situates each component in its anatomical context and presents the complete model in a single view. Components are placed across brain regions and connected into a working circuit through direct manipulation. A mini-brain overview, along with resizable and collapsible panels, maintains a clear workspace as models become more complex.

## A comprehensive library of dynamical blox

The **component library**, presented along the bottom of the workspace, provides the full catalog of Neuroblox building blocks, organized by category and available for direct placement onto the canvas. The library spans multiple scales of biological detail:

- **Individual neurons** — Hodgkin-Huxley and integrate-and-fire formulations (IF, QIF, LIF)
- **Neural mass models** — Jansen-Rit, Wilson-Cowan, and Next-Generation
- **Neural assemblies and composite blox** — multi-scale, biomimetic micro-circuits composed of neurons, synapses, and micro-assemblies
- **Stimulus sources** — sensory, experimental, and task-driven inputs to the circuit

## Biologically grounded connectivity

Connections between brain regions represent more than an abstract graph edge. Each connection is defined by its **strength** and by the **specific neurotransmitters and receptors** that dominate the pathway, so that models reflect the underlying signaling biology. Neuroblox supports an extensive set of receptor dynamics, including glutamatergic AMPA and NMDA, GABA-A and GABA-B, dopaminergic D1 and D2, serotonergic 5-HT, cholinergic nicotinic and muscarinic acetylcholine, and TRPM4. This resolution is central to modeling receptor-level and pharmacological interventions.

## Detailed parameterization

All model parameters are configured in the **left sidebar**. Selecting any blox or connection exposes its full set of properties — intrinsic dynamics, receptor and synaptic settings, connection weights, and related parameters — at the level of detail otherwise available only through the programmatic interface. A node list controls which blox outputs are recorded for analysis, with a corresponding indicator on the canvas.

## Simulation and analysis

Completed circuits are simulated directly within the environment. Simulation control, output selection, category-learning tasks, and experimental parameters are configured in the **right sidebar**. From this panel, users run simulations and review results inline or in an expanded canvas view, define experiments, fit model parameters to experimental data (including electrophysiology, neuroimaging, and behavioral measures), and configure reinforcement-learning and category-learning tasks with dedicated result visualization. Models compile to high-performance numerical kernels, making large parameter sweeps and optimization studies practical.

## Consistency with the Neuroblox programming interface

NeurobloxGUI is designed to provide the full modeling capability of the Neuroblox platform through a visual interface. The great majority of workflows available in Neuroblox code — the same blox, receptors, connections, experiments, and parameter-fitting procedures documented in the [Neuroblox documentation](https://neuroblox.github.io/NeurobloxDocsHost/dev/) — can be carried out entirely in the GUI. As this is a beta release, complete parity with the programmatic interface is being finalized and will be delivered in the 0.95 release. Circuits can be imported and exported (including via the Neuroblox DSL), allowing visual and programmatic workflows to interoperate.

## Availability and licensing

Neuroblox is free for non-commercial and academic use under the Neuroblox EULA. For commercial and pharmaceutical use, please contact Neuroblox at <info@neuroblox.ai>. Documentation and installation instructions are available at [neuroblox.github.io/NeurobloxDocsHost](https://neuroblox.github.io/NeurobloxDocsHost/dev/).

> [!WARNING]
> This is a beta software release. Features and interfaces are subject to change prior to the 1.0 release.# Neuroblox.jl

| **Documentation** | **Citation** |
|:-------------------------------------------------------------------------------:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] |  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14192732.svg)](https://doi.org/10.5281/zenodo.14192732)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml

