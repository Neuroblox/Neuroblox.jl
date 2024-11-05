# # Synaptic plasticity tutorial


using Neuroblox 
using OrdinaryDiffEq ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Random ## for generating random variables
using CairoMakie ## for customized plotting recipies for blox
using CSV ## to read data from CSV files
using DataFrames ## to format the data into DataFrames
using Downloads ## to download image stimuli files


# ## Cortico-cortical plasticity

time_block_dur = 90.0 ## ms (size of discrete time blocks)
N_trials = 5 ##number of trials
trial_dur = 1000 ##ms

# create an image source block which takes image data from a .csv file and gives input to visual cortex

image_set = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv"), DataFrame) ## reading data into DataFrame format
## change the source file to stimuli_set.csv

global_namespace=:g
## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
@named stim = ImageStimulus(image_set[1:N_trials,:]; namespace=global_namespace, t_stimulus=trial_dur, t_pause=0); 

## cortical blox
@named VAC = CorticalBlox(N_wta=4, N_exci=5,  density=0.05, weight=1,I_bg_ar=0;namespace=global_namespace) 
@named AC = CorticalBlox(N_wta=2, N_exci=5, density=0.05, weight=1,I_bg_ar=0;namespace=global_namespace) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEIBlox(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

## definelearning rule
hebbian_cort = HebbianPlasticity(K=5e-3, W_lim=7, t_pre=trial_dur-eps(), t_post=trial_dur-eps()) 
	

g = MetaDiGraph()

add_edge!(g, stim => VAC, weight=14) 
add_edge!(g, ASC1 => VAC, weight=44)
add_edge!(g, ASC1 => AC, weight=44)
add_edge!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort) ## give learning rule as parameter


agent = Agent(g; name=:ag, t_block = time_block_dur); ## define agent
env = ClassificationEnvironment(stim; name=:env, namespace=global_namespace)
run_experiment!(agent, env)#; alg=Vern7())


# ## cortico-striatal circuit performing category learning 


time_block_dur = 90.0 ## ms (size of discrete time blocks)
N_trials = 10 ##number of trials
trial_dur = 600 ##ms

# create an image source block which takes image data from a .csv file and gives input to visual cortex

image_set = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv"), DataFrame) ## reading data into DataFrame format
## change the source file to stimuli_set.csv

global_namespace=:g
## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
@named stim = ImageStimulus(image_set[1:N_trials,:]; namespace=global_namespace, t_stimulus=trial_dur, t_pause=0); 

## cortical blox
@named VAC = CorticalBlox(N_wta=4, N_exci=5,  density=0.05, weight=1,I_bg_ar=0;namespace=global_namespace) 
@named AC = CorticalBlox(N_wta=2, N_exci=5, density=0.05, weight=1,I_bg_ar=0;namespace=global_namespace) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEIBlox(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

## striatum blocks
@named STR1 = Striatum(N_inhib=5;namespace=global_namespace) 
@named STR2 = Striatum(N_inhib=5;namespace=global_namespace) 

@named tan_pop1 = TAN(;namespace=global_namespace) 
@named tan_pop2 = TAN(;namespace=global_namespace) 
	
@named AS = GreedyPolicy(namespace=global_namespace, t_decision=2*time_block_dur+eps()) 
@named SNcb = SNc(namespace=global_namespace) 


hebbian_mod = HebbianModulationPlasticity(K=0.04, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=trial_dur-eps(), t_post=trial_dur-eps(), t_mod=time_block_dur)
hebbian_cort = HebbianPlasticity(K=5e-3, W_lim=7, t_pre=trial_dur-eps(), t_post=trial_dur-eps()) 

## circuit 



g = MetaDiGraph()

add_edge!(g, stim => VAC, weight=14) 
add_edge!(g, ASC1 => VAC, weight=44)
add_edge!(g, ASC1 => AC, weight=44)
add_edge!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort) 
add_edge!(g, tan_pop1 => STR1, weight = 1, t_event = time_block_dur)
add_edge!(g, tan_pop2 => STR2, weight = 1, t_event = time_block_dur)
add_edge!(g, STR1 => tan_pop1, weight = 1)
add_edge!(g, STR2 => tan_pop1, weight = 1)
add_edge!(g, STR1 => tan_pop2, weight = 1)
add_edge!(g, STR2 => tan_pop2, weight = 1)
add_edge!(g, STR1 => STR2, weight = 1, t_event = 2*(time_block_dur+eps()))
add_edge!(g, STR2 => STR1, weight = 1, t_event = 2*(time_block_dur+eps()))
add_edge!(g, STR1 => AS)
add_edge!(g, STR2 => AS)
add_edge!(g, STR1 => SNcb, weight = 1) 
add_edge!(g, STR2 => SNcb, weight = 1)  


agent = Agent(g; name=:ag, t_block = time_block_dur); ## define agent
env = ClassificationEnvironment(stim; name=:env, namespace=global_namespace)
run_experiment!(agent, env)#;