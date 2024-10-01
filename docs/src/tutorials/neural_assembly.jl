# # Tutorial on bottom-up approach to building a neural assembly


# ## Single spiking neuron from Hodgkin-Huxley model

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
using MetaGraphs  ## use its MetaGraph type to build the circuit
using DifferentialEquations ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Random ## for generating random variables
using Plots ## Plotting timeseries
using CairoMakie ## for customized plotting recipies for blox
using CSV ## to read data from CSV files
using DataFrames ## to format the data into DataFrames

# define a single excitatory neuron 'blox' with steady input current I_bg = 0.5 microA/cm2
nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=0.5)

# add the single neuron 'blox' as a single node in a graph
g = MetaDiGraph() ## defines a graph
add_blox!.(Ref(g), [nn1]) ## adds the defined blocks into the graph


# create an ODESystem from the graph
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
length(unknowns(sys)) ## shows the number of variables in the simplified system

# To solve the system, we first create an Ordinary Differential Equation Problem and then solve it over the tspan of (0,1e) using a Vern7() solver.  The solution is saved every 0.1ms. The unit of time in Neuroblox is 1ms.


prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

# acessing the voltage timeseries from the neuron block and plotting the voltage
v = voltage_timeseries(n_inh,sol)
Plots.plot(sol.t,v,xlims=(0,1000),xlabel="time (ms)",ylabel="mV")


# ## Connecting three neurons through synapses to make a small local circuit

## While creating a system of multiple components (neurons in this case), each component should be defined within the same namespace. So first
## we define a global namespace.
global_namespace=:g

## define three neurons, two excitatory and one inhibitory 

nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_bg=0.4,namespace=global_namespace)
nn2 = HHNeuronInhibBlox(name=Symbol("nrn2"), I_bg=0.1,namespace=global_namespace)
nn3 = HHNeuronExciBlox(name=Symbol("nrn3"), I_bg=1.4,namespace=global_namespace)
assembly = [nn1,nn2,nn3] 

## add the three neurons as nodes in a graph
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)

## connect the nodes with the edges (synapses in this case), with the synaptic weights specified as arguments
add_edge!(g, 1, 2, Dict(:weight => 1)) ##connection from node 1 to node 2 (nn1 to nn2)
add_edge!(g, 2, 3, Dict(:weight => 0.2)) ##connection from node 2 to node 3 (nn2 to nn3)

## create an ODESystem from the graph and then solve it using an ODE solver
@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

## plotting membrane voltage activity of all neurons in a stacked form

voltage_stack([nn1,nn2,nn3],sol)	## voltage_stack(<blox or array of blox>, sol)



# ## Creating a lateral inhibition circuit (the "winner-takes-all" circuit) in superficial cortical layer

global_namespace=:g 
N_exci = 5; ##number of excitatory neurons

n_inh = HHNeuronInhibBlox(name = Symbol("inh"), namespace=global_namespace, G_syn = 4.0, τ = 70) ##feedback inhibitory interneuron neuron

##creating an array of excitatory pyramidal neurons
n_excis = [HHNeuronExciBlox(
                            name = Symbol("exci$i"),
                            namespace=global_namespace, 
                            G_syn = 3.0, 
                            τ = 5,
                            I_bg = 5*rand(), 
                            ) for i = 1:N_exci]

g = MetaDiGraph()
add_blox!(g, n_inh)
for i in Base.OneTo(N_exci)
    add_blox!(g, n_excis[i])
    add_edge!(g, 1, i+1, :weight, 1.0)
    add_edge!(g, i+1, 1, :weight, 1.0)
end

@named sys = system_from_graph(g)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)
voltage_stack(n_excis,sol)

# ## Creating lateral inhibition "winner-take-all" circuit (WTA) blocks from the inbuilt functions and connecting two WTA circuit blocks


global_namespace=:g
N_exci = 5 ##number of excitatory neurons in each WTA circuit
wta1 = WinnerTakeAllBlox(name=Symbol("wta1"), I_bg=5.0, N_exci=N_exci, namespace=global_namespace) ##for a single valued input current, each neuron of the WTA circuit will recieve a uniformly distributed random input from 0 to I_bg  
wta2 = WinnerTakeAllBlox(name=Symbol("wta2"), I_bg=4.0, N_exci=N_exci, namespace=global_namespace)

g = MetaDiGraph()
add_blox!.(Ref(g), [wta1, wta2])
add_edge!(g, 1, 2, Dict(:weight => 1, :density => 0.5)) ##density keyword sets the connection probability from each excitatory neuron of source WTA circuit to each excitatory neuron of target WTA circuit

sys = system_from_graph(g, name=global_namespace)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

voltage_stack([wta1,wta2],sol)

# ## Creating a single cortical superficial layer block (SCORT in Pathak et. al. 2024) by connecting multiple WTA circuits

global_namespace=:g 
N_wta=20 ## number of WTA circuits
## parameters
N_exci=5   ##number of pyramidal neurons in each lateral inhibition (WTA) circuit
G_syn_exci=3.0 ##maximal synaptic conductance in glutamatergic (excitatory) synapses
G_syn_inhib=4.0 ## maximal synaptic conductance in GABAergic (inhibitory) synapses from feedback interneurons
G_syn_ff_inhib=3.5 ## maximal synaptic conductance in GABAergic (inhibitory) synapses from feedforward interneurons
I_bg=5.0 ##background input
density=0.01 ##connection density between WTA circuits

##creating array of WTA ciruits
wtas = [WinnerTakeAllBlox(;
                           name=Symbol("wta$i"),
                           namespace=global_namespace,
                           N_exci=N_exci,
                           G_syn_exci=G_syn_exci,
                           G_syn_inhib=G_syn_inhib,
                           I_bg = I_bg  
                          ) for i = 1:N_wta]

##feed-forward interneurons (get input from other pyramidal cells and from the ascending system, largely controls the rhythm)
n_ff_inh = HHNeuronInhibBlox(;
                             name=Symbol("ff_inh"),
                             namespace=global_namespace,
                             G_syn=G_syn_ff_inhib
                            )

g = MetaDiGraph()
add_blox!.(Ref(g), vcat(wtas, n_ff_inh))

## connecting WTA circuits to each other with given connection density, and feedforward interneuron connects to each WTA circuit 
for i in 1:N_wta
    for j in 1:N_wta
        if j != i
            add_edge!(g, i, j, Dict(:weight => 1, :density => density))
        end
    end
    add_edge!(g, N_wta+1, i, Dict(:weight => 1))
end

sys = system_from_graph(g, name=global_namespace)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), [])
sol = solve(prob, Vern7(), saveat=0.1)

voltage_stack(vcat(wtas, n_ff_inh),sol)


# ## Creating an ascending system block (ASC1 in Pathak et. al. 2024), a single inbuilt cortical superficial layer block (SCORT in Pathak et. al. 2024) and connecting them.

global_namespace=:g

## define ascending system block using a Next Generation Neural Mass model as described in Byrne et. al. 2020.
## the parameters are fixed to generate a 16 Hz modulating frequency in the cortical neurons  
@named ASC1 = NextGenerationEIBlox(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

## define the superficial layer cortical block using inbuilt function
## Number if WTA circuits = N_wta=45; number of pyramidal neurons in each WTA circuit = N_exci = 5;
@named CB = CorticalBlox(N_wta=45, N_exci=5, density=0.01, weight=1,I_bg_ar=5;namespace=global_namespace)

## define graph and add both blox in the graph
g = MetaDiGraph()
add_blox!.(Ref(g), [ASC1,CB])
## connect ASC1->CB
add_edge!(g, 1, 2, Dict(:weight => 44))

## solve the system for time 0 to 1000 ms
sys = system_from_graph(g, name=global_namespace)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), []) ## tspan = (0,1000)
sol = solve(prob, Vern7(), saveat=0.1)

# plot neuron time series
voltage_stack(CB,sol)

# plot the meanfield of all cortical block neurons (mean membrane voltage)
mnv = meanfield_timeseries(CB,sol)
Plots.plot(sol.t,mnv,xlabel="time (ms)",ylabel="mV") 

# plot power spectrum of the meanfield
powerspectrumplot(CB,sol)

# Notice the peak at 16 Hz, representing beta oscillations.


# ## Creating simulation of visual stimulus response in cortical blocks (add Hebbian learning later)

# create cortical blocks for visual area cortex (VAC), anterior cortex (AC) and ascending system block (ASC1)
global_namespace=:g
## cortical blox
@named VAC = CorticalBlox(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0;namespace=global_namespace) 
@named AC = CorticalBlox(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0;namespace=global_namespace) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEIBlox(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

# create an image source block which takes image data from a .csv file and gives input to visual cortex

fn = joinpath(@__DIR__, "../data/image_example.csv") ## image data file
image_set = CSV.read(fn, DataFrame) ## reading data into DataFrame format
image_sample = 10 ## set which image to input (from 1 to 1000)

## plot the image that the visual cortex 'sees'
pixels=Array(image_set[image_sample,1:end-1])## access the desired image sample from respective row
pixels=reshape(pixels,15,15)## reshape into 15 X 15 square image matrix
Plots.plot(Gray.(ar),xlims=(0.5,15.5),ylims=(0.5,15.5)) ## plot the grayscale image

## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
## to try new image samples, change the image_sample and re-run the subsequent code lines 
@named stim = ImageStimulus(image_set[[image_sample],:]; namespace=global_namespace, t_stimulus=1000, t_pause=0) 

# assemble the blox into a graph and set connections

circuit = [stim,VAC,AC,ASC1]
d = Dict(b => i for (i,b) in enumerate(circuit)) ## can refer to nodes through their names instead of their indices

g = MetaDiGraph()
add_blox!.(Ref(g), circuit) ## add all the blox into the graph

## set connections and their keword arguments like connection weight and connection density
add_edge!(g, d[stim], d[VAC], :weight, 14) 
add_edge!(g, d[ASC1], d[VAC], Dict(:weight => 44))
add_edge!(g, d[ASC1], d[AC], Dict(:weight => 44))
add_edge!(g, d[VAC], d[AC], Dict(:weight => 3, :density => 0.08))

## define odesyste, simplify and solve
sys = system_from_graph(g, name=global_namespace)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 1000), []) ## tspan = (0,1000)
sol = solve(prob, Vern7(), saveat=0.1)

##plot voltage stacks, mean fields and powerspectrums

## VAC
voltage_stack(VAC,sol)
mnv = meanfield_timeseries(VAC,sol)
Plots.plot(sol.t,mnv)
powerspectrumplot(VAC,sol)

## AC
voltage_stack(AC,sol)
mnv = meanfield_timeseries(AC,sol)
Plots.plot(sol.t,mnv)
powerspectrumplot(AC,sol)