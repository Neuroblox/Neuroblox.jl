# # Introduction

# This tutorial provides a simple example of how to use the Neuroblox package to simulate a pyramidal-interneuron gamma (PING) network. 
# These networks are generally useful in modeling cortical oscillations and are used in a variety of contexts.
# This particular example is based on Börgers, Epstein, and Kopell (2005) and is a simple example of how to replicate their initial network in Neuroblox.

# # PING network - conceptual definition
# The PING network is a simple model of a cortical network that consists of two populations of neurons: excitatory and inhibitory.
# 

# # Import the necessary packages
## Reasons for each non-Neuroblox package are given in the comments after each
using Neuroblox 
using DifferentialEquations ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Distributions ## for statistical distributions 
using MetaGraphs ## use its MetaGraph type to build the circuit
using Random ## for random number generation
using CairoMakie ## for plotting

Random.seed!(42)

# Setup parameters from the supplemental material
μ_E = 0.8
σ_E = 0.15
μ_I = 0.8
σ_I = 0.08

# Define the PING network neuron numbers
NE_driven = 40
NE_other = 120
NI_driven = 40
N_total = NE_driven + NE_other + NI_driven

# Define the extra currents
I_driveE = Normal(μ_E, σ_E)
I_driveI = Normal(μ_I, σ_I)
I_base = Normal(0, 0.1)
I_undriven = Normal(0, 0.4)
I_bath = -0.7

# First, create the 20 driven excitatory neurons
exci_driven = [PINGNeuronExci(name=Symbol("ED$i"), I_ext=rand(I_driveE)+rand(I_base)) for i in 1:NE_driven]
exci_other  = [PINGNeuronExci(name=Symbol("EO$i"), I_ext=rand(I_base) + rand(I_undriven)) for i in 1:NE_other]
exci        = [exci_driven; exci_other]
inhib       = [PINGNeuronInhib(name=Symbol("ID$i"), I_ext=rand(I_driveI) + rand(I_base)+I_bath) for i in 1:NI_driven]
# Create the network
g = MetaDiGraph()
add_blox!.(Ref(g), vcat(exci, inhib))

# Extra parameters
N=N_total
g_II=0.2 
g_IE=0.6 
g_EI=0.8

for ne ∈ exci
    for ni ∈ inhib
        add_edge!(g, ne => ni; weight=g_EI/N)
        add_edge!(g, ni => ne; weight=g_IE/N)
    end
end

for ni1 ∈ inhib
    for ni2 ∈ inhib
        add_edge!(g, ni1 => ni2; weight=g_II/N)
    end
end

@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], (0.0, 300.0))
@time sol = solve(prob, Tsit5(), saveat=0.1)

rasterplot(exci,  sol; threshold=20.0)
rasterplot(inhib, sol; threshold=20.0)

# # References
# 1. Börgers C, Epstein S, Kopell NJ. Gamma oscillations mediate stimulus competition and attentional selection in a cortical network model. 
# Proc Natl Acad Sci U S A. 2008 Nov 18;105(46):18023-8. DOI: [10.1073/pnas.0809511105](https://www.doi.org/10.1073/pnas.0809511105). Epub 2008 Nov 12. PMID: 19004759; PMCID: PMC2584712.