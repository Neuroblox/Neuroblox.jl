# # Bottom-up construction of a neural assembly

# ## Introduction
# This tutorial goes through the process of building a neural assembly that is part of a larger model that performs category learning of images [1]. We will follow a bottom-up approach with these steps :
# - build a model of a single neuron
# - expand that model by connecting a few neurons into a local circuit
# - define a "winner-takes-all" (WTA) circuit with lateral inhibition
# - build a cortical block by connecting multiple WTAs together with feed-forward inhibition 
# - connect the cortical block to a model of an ascending system
# - add a source of visual input (images) and a cortical block representing visual cortex to our model and simulate visual processing

# ## Single spiking neuron from Hodgkin-Huxley model
# ![fig1](../assets/neural_assembly_1.png)

# Hodgkin-Huxley (HH) formalism to describe membrane potential of a single neuron

 # ```math
 #   \begin{align}
 #   C_m\frac{dV}{dt} &= -g_L(V-V_L) - {g}_{Na}m^3h(V-V_{Na}) -{g}_Kn^4(V-V_K) + I_{in} - I_{syn} \\
 #   \frac{dm}{dt} &= \alpha_{m}(V)(1-m) + \beta_{m}(V)m \\ 
 #   \frac{dh}{dt} &= \alpha_{h}(V)(1-h) + \beta_{h}(V)h \\
 #   \frac{dn}{dt} &= \alpha_{n}(V)(1-n) + \beta_{n}(V)n 
 #   \end{align}
 # ```


using Neuroblox
using OrdinaryDiffEqVerner ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Random ## for generating random variables
using CairoMakie ## for customized plotting recipies for blox
using CSV ## to read data from CSV files
using DataFrames ## to format the data into DataFrames
using Downloads ## to download image stimuli files

# define a single excitatory neuron 'blox' with steady input current I_bg = 0.5 microA/cm2
nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=0.5)

# define graph and add the single neuron 'blox' as a single node into the graph
g = GraphSystem() ## defines a graph
add_node!(g, nn1) ## adds the defined blocks into the graph

# To solve the system, we first create an Ordinary Differential Equation Problem and then solve it over the tspan of (0,1e) using a Vern7() solver.  The solution is saved every 0.1ms. The unit of time in Neuroblox is 1ms.

prob = ODEProblem(g, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1);

# acessing the voltage timeseries from the neuron block and plotting the voltage
v = voltage_timeseries(nn1, sol)

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
cl = get_neuron_color(nn1) #specify color based on neuron type (excitatory/inhibitory)
lines!(ax, sol.t, v, color=cl)
fig ## to display the figure

# Suggestion : Try different values of input current 'I_bg' and run the entire code block to see the output activity
# ## Connecting three neurons through synapses to make a local feed-forward circuit

# ![fig2](../assets/neural_assembly_2.png)

## While creating a system of multiple components (neurons in this case), each component should be defined within the same namespace. So first
## we define a global namespace.
global_namespace=:g
## define three neurons, two excitatory and one inhibitory 

nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=0.4, namespace=global_namespace)
nn2 = HHNeuronInhib(name=Symbol("nrn2"), I_bg=0.1, namespace=global_namespace)
nn3 = HHNeuronExci(name=Symbol("nrn3"), I_bg=1.4, namespace=global_namespace)

## defien graph and connect the nodes with the edges (synapses in this case), with the synaptic 'weights' specified as arguments
g = GraphSystem(name=global_namespace)
add_connection!(g, nn1 => nn2, weight = 1) ##connection from neuron 1 to neuron 2 (nn1 to nn2)
add_connection!(g, nn2 => nn3, weight = 0.2) ##connection from neuron 2 to neuron 3 (nn2 to nn3)
add_connection!(g, nn1 => nn3, weight = 0.5) ##connection from neuron 1 to neuron 3 (nn2 to nn3)

## create an ODESystem from the graph and then solve it using an ODE solver
prob = ODEProblem(g, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1);

## plotting membrane voltage activity of all neurons in a stacked form

stackplot([nn1,nn2,nn3], sol)	## stackplot(<blox or array of blox>, sol)

# Suggestion : Try different values of input currents 'I_bg' and connection weights. One can try different permutations of excitatory and inhibitory neurons.

# ## Creating a lateral inhibition circuit (the "winner-takes-all" circuit) in superficial cortical layer

# ![fig3](../assets/neural_assembly_3.png)

global_namespace=:g 
N_exci = 5; ##number of excitatory neurons

n_inh = HHNeuronInhib(name = Symbol("inh"), namespace=global_namespace, G_syn = 4.0, τ = 70) ##feedback inhibitory interneuron neuron

##creating an array of excitatory pyramidal neurons
n_excis = [HHNeuronExci(
    name = Symbol("exci$i"),
    namespace=global_namespace, 
    G_syn = 3.0, 
    τ = 5,
    I_bg = 5*rand(), 
) for i = 1:N_exci]

g = GraphSystem()

for i in 1:N_exci
    add_connection!(g, n_inh => n_excis[i], weight = 1.0)
    add_connection!(g, n_excis[i] => n_inh, weight = 1.0)
end
prob = ODEProblem(g, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
stackplot(vcat(n_excis, n_inh), sol)

# Suggestion : Instead of uniform random input current in each excitatory neuron, try different configurations (random or constant) of input currents I_bg for each neuron. 
# One can vary the size of circuit by changing number of excitatory neurons. 

# ## Creating lateral inhibition "winner-take-all" circuit (WTA) blocks from the inbuilt functions and connecting two WTA circuit blocks


global_namespace=:g
N_exci = 5 ##number of excitatory neurons in each WTA circuit
wta1 = WinnerTakeAll(name=Symbol("wta1"), I_bg=5.0, N_exci=N_exci, namespace=global_namespace) ##for a single valued input current, each neuron of the WTA circuit will recieve a uniformly distributed random input from 0 to I_bg  
wta2 = WinnerTakeAll(name=Symbol("wta2"), I_bg=4.0, N_exci=N_exci, namespace=global_namespace)

g = GraphSystem()
add_connection!(g, wta1 => wta2, weight=1, density=0.5) ##density keyword sets the connection probability from each excitatory neuron of source WTA circuit to each excitatory neuron of target WTA circuit

prob = ODEProblem(g, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

neuron_set = get_neurons([wta1, wta2]) ## extract neurons from a composite blocks 
stackplot(neuron_set,sol)

# ## Creating a single cortical superficial layer block by connecting multiple WTA circuits

# This model is SCORT in [1] and looks like this
# ![fig4](../assets/neural_assembly_4.png)

global_namespace=:g 
N_wta=10 ## number of WTA circuits
## parameters
N_exci=5   ##number of pyramidal neurons in each lateral inhibition (WTA) circuit
G_syn_exci=3.0 ##maximal synaptic conductance in glutamatergic (excitatory) synapses
G_syn_inhib=4.0 ## maximal synaptic conductance in GABAergic (inhibitory) synapses from feedback interneurons
G_syn_ff_inhib=3.5 ## maximal synaptic conductance in GABAergic (inhibitory) synapses from feedforward interneurons
I_bg=5.0 ##background input
density=0.01 ##connection density between WTA circuits

##creating array of WTA ciruits
wtas = [WinnerTakeAll(;
                      name=Symbol("wta$i"),
                      namespace=global_namespace,
                      N_exci=N_exci,
                      G_syn_exci=G_syn_exci,
                      G_syn_inhib=G_syn_inhib,
                      I_bg = I_bg  
                      ) for i = 1:N_wta]

##feed-forward interneurons (get input from other pyramidal cells and from the ascending system, largely controls the rhythm)
n_ff_inh = HHNeuronInhib(;
                         name=Symbol("ff_inh"),
                         namespace=global_namespace,
                         G_syn=G_syn_ff_inhib
                         )

g = GraphSystem()

## connecting WTA circuits to each other with given connection density, and feedforward interneuron connects to each WTA circuit 
for i in 1:N_wta
    for j in 1:N_wta
        if j != i
            add_connection!(g, wtas[i] => wtas[j], weight=1, density=density)
        end
    end
    add_connection!(g, n_ff_inh => wtas[i], weight=1)
end

prob = ODEProblem(g, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

neuron_set = get_neurons(vcat(wtas, n_ff_inh)) ## extract neurons from a composite blocks
stackplot(neuron_set, sol)

# Sugestion : try different connection densities and weights and see how it affects the population activity. 

# ## Connecting the cortical superficial layer block to an ascending system block

# Now we will expand on the SCORT block of the previous section by defining a block representing an ascending system (ASC1 in [1]) and then connecting the two blocks together. 

global_namespace=:g

## define ascending system block using a Next Generation Neural Mass model as described in Byrne et. al. 2020.
## the parameters are fixed to generate a 16 Hz modulating frequency in the cortical neurons  
@named ASC1 = NextGenerationEI(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

## define the superficial layer cortical block using inbuilt function
## Number if WTA circuits = N_wta=45; number of pyramidal neurons in each WTA circuit = N_exci = 5;
@named CB = Cortical(N_wta=10, N_exci=5, density=0.01, weight=1, I_bg_ar=7; namespace=global_namespace)

## define graph and connect ASC1->CB
g = GraphSystem()
add_connection!(g, ASC1 => CB, weight=44)

## solve the system for time 0 to 1000 ms

prob = ODEProblem(g, [], (0.0, 1000), []) ## tspan = (0,1000)
sol = solve(prob, Vern7(), saveat=0.1);

# plot neuron time series
neuron_set = get_neurons(CB) ## extract neurons from a composite block like Cortical
n_neurons = 50 ## set number nof neurons to display in the stackplot
stackplot(neuron_set[1:n_neurons], sol)

# plot the meanfield of all cortical block neurons (mean membrane voltage)
mnv = meanfield_timeseries(CB, sol)
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Meanfield voltage (mv)")
lines!(ax, sol.t, mnv)
fig ## to display the figure

# plot power spectrum of the meanfield (average over membrane potentials)
powerspectrumplot(CB, sol)

# Notice the peak at 16 Hz, representing beta oscillations.
# Sugestion : try changing parameters of ASC1 to generate different cortical rhythms. See how the peak shifts in the powerspectrum
# ## Creating simulation of visual stimulus response in cortical blocks 

# ![fig5](../assets/neural_assembly_5.png)
# create cortical blocks for visual area cortex (VAC), anterior cortex (AC) and ascending system block (ASC1)
global_namespace=:g
## cortical blox
@named VAC = Cortical(N_wta=10, N_exci=5,  density=0.01, weight=1,I_bg_ar=0; namespace=global_namespace) 
@named AC = Cortical(N_wta=10, N_exci=5, density=0.01, weight=1,I_bg_ar=0; namespace=global_namespace) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEI(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

# create an image source block which takes image data from a .csv file and gives input to visual cortex

image_set = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/image_example.csv"), DataFrame) ## reading data into DataFrame format
image_sample = 2 ## set which image to input (from 1 to 1000)

## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
@named stim = ImageStimulus(image_set[[image_sample], :]; namespace=global_namespace, t_stimulus=1000, t_pause=0); 


# plot the image that the visual cortex 'sees'
pixels = Array(image_set[image_sample, 1:end-1])## access the desired image sample from respective row
pixels = reshape(pixels, 15, 15)## reshape into 15 X 15 square image matrix
heatmap(pixels,colormap = :gray1) #input image matrix seen as heatmap

# assemble the blox into a graph and set connections with their keword arguments like connection weight and connection density

g = GraphSystem()

add_connection!(g, stim => VAC, weight=14) 
add_connection!(g, ASC1 => VAC, weight=44)
add_connection!(g, ASC1 => AC, weight=44)
add_connection!(g, VAC => AC, weight=3, density=0.08)

## define system and solve

prob = ODEProblem(g, [], (0.0, 1000), []) ## tspan = (0,1000)
sol = solve(prob, Vern7(), saveat=0.1);

# Let us now plot neuron potentials, meanfield activity and powerspectrums for the VAC and AC blox.
# First we show the stackplot of voltage potentials from the first 10 neurons of VAC
VAC_neuron_set = get_neurons(VAC) ## extract neurons from VAC
n_neurons = 40 ##number of neurons displayed. You can try incresing it.
stackplot(VAC_neuron_set[1:n_neurons],sol)

# then we plot the meanfield potential out of all neurons within VAC
mnv = meanfield_timeseries(VAC, sol)

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
lines!(ax, sol.t, mnv)
fig ## to display the figure

# Here is the powerspectrum from all neurons within VAC
powerspectrumplot(VAC, sol)

# Moving on to the AC blox, we first plot the voltage potential of its neurons
AC_neuron_set = get_neurons(AC) ## extract neurons from VAC
n_neurons = 40
stackplot(AC_neuron_set[1:n_neurons], sol)

# followed by the meanfield activity 
mnv = meanfield_timeseries(AC, sol)
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Voltage (mv)")
lines!(ax, sol.t, mnv)
fig ## to display the figure

# and finally the AC powerspectrum
powerspectrumplot(AC, sol)

# Sugestion : Try changing the image samples and notice the change in the spatial firing patterns in VAC and AC neurons. One can make multiple cortical blocks simillar to AC and connect them in various 
# connection topologies. All of them can directly or indirectly get input from VAC.  

# ## References
# [1] Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun (2025). https://doi.org/10.1038/s41467-025-67076-x
