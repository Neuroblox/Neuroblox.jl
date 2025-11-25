# # Synaptic plasticity and Reinforcement Learning

using Neuroblox 
using OrdinaryDiffEqDefault, OrdinaryDiffEqVerner ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Random ## for generating random variables
using CairoMakie ## for customized plotting recipies for blox
using CSV ## to read data from CSV files
using DataFrames ## to format the data into DataFrames
using Downloads ## to download image stimuli files

# ## Cortico-cortical plasticity

N_trials = 5 ##number of trials
trial_dur = 1000 ##ms

# create an image source block which takes image data from a .csv file and gives input to visual cortex

image_set = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv"), DataFrame) ## reading data into DataFrame format
## change the source file to stimuli_set.csv
 
model_name=:g
## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
@named stim = ImageStimulus(image_set; namespace=model_name, t_stimulus=trial_dur, t_pause=0); 

## cortical blox
@named VAC = Cortical(; namespace=model_name, N_wta=4, N_exci=5,  density=0.05, weight=1) 
@named AC = Cortical(; namespace=model_name, N_wta=2, N_exci=5, density=0.05, weight=1) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEI(; namespace=model_name, Cₑ=2*26,Cᵢ=1*26, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑᵢ=0.6*26, kᵢₑ=0.6*26) 

## define learning rule
hebbian_cort = HebbianPlasticity(K=5e-5, W_lim=7, t_pre=trial_dur, t_post=trial_dur) 
	
g = MetaDiGraph()

add_edge!(g, stim => VAC, weight=14) 
add_edge!(g, ASC1 => VAC, weight=44)
add_edge!(g, ASC1 => AC, weight=44)
add_edge!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort) ## give learning rule as parameter

agent = Agent(g; name=model_name); ## define agent
env = ClassificationEnvironment(stim, N_trials; name=:env, namespace=model_name)

fig = Figure(title="Adjacency matrix", size = (1600, 800))

adjacency(fig[1,1], agent)

run_experiment!(agent, env; t_warmup=200.0, alg=Vern7(), verbose=true)

adjacency(fig[1,2], agent)

fig

# ## Cortico-striatal circuit performing category learning 

# ##  It is one simplified biological instantiation of a reinforcement-learning system; 
# ##  It is carrying out simple RL learning behavior but not faithfully simulating physiology. 
# ## The experiment it is trying to simulate is the category learning experiment [Antzoulatos2014] which was successfully modeled through a detailed corticostriatal model (Pathak et. al.)


time_block_dur = 90.0 ## ms (size of discrete time blocks)
N_trials = 100 ##number of trials
trial_dur = 1000 ##ms

# create an image source block which takes image data from a .csv file and gives input to visual cortex

image_set = CSV.read(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv"), DataFrame) ## reading data into DataFrame format
## change the source file to stimuli_set.csv

model_name=:g
## define stimulus source blox
## t_stimulus: how long the stimulus is on (in msec)
## t_pause : how long th estimulus is off (in msec)
@named stim = ImageStimulus(image_set; namespace=model_name, t_stimulus=trial_dur, t_pause=0); 

## cortical blox
@named VAC = Cortical(; namespace=model_name, N_wta=4, N_exci=5,  density=0.05, weight=1) 
@named AC = Cortical(; namespace=model_name, N_wta=2, N_exci=5, density=0.05, weight=1) 
## ascending system blox, modulating frequency set to 16 Hz
@named ASC1 = NextGenerationEI(; namespace=model_name, Cₑ=2*26,Cᵢ=1*26, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑᵢ=0.6*26, kᵢₑ=0.6*26) 

## striatum blocks
@named STR1 = Striatum(; namespace=model_name, N_inhib=5) 
@named STR2 = Striatum(; namespace=model_name, N_inhib=5) 

@named tan_pop1 = TAN(κ=10; namespace=model_name) 
@named tan_pop2 = TAN(κ=10; namespace=model_name) 
	
@named AS = GreedyPolicy(; namespace=model_name, t_decision=2*time_block_dur) 
@named SNcb = SNc(κ_DA=1; namespace=model_name) 

hebbian_mod = HebbianModulationPlasticity(K=0.05, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=trial_dur, t_post=trial_dur, t_mod=time_block_dur)
hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=7, t_pre=trial_dur, t_post=trial_dur) 

## circuit 

g = MetaDiGraph()

add_edge!(g, stim => VAC, weight=14) 
add_edge!(g, ASC1 => VAC, weight=44)
add_edge!(g, ASC1 => AC, weight=44)
add_edge!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort) 
add_edge!(g, AC=>STR1, weight = 0.075, density =  0.04, learning_rule =  hebbian_mod)
add_edge!(g, AC=>STR2, weight =  0.075, density =  0.04, learning_rule =  hebbian_mod) 
add_edge!(g, tan_pop1 => STR1, weight = 1, t_event = time_block_dur)
add_edge!(g, tan_pop2 => STR2, weight = 1, t_event = time_block_dur)
add_edge!(g, STR1 => tan_pop1, weight = 1)
add_edge!(g, STR2 => tan_pop1, weight = 1)
add_edge!(g, STR1 => tan_pop2, weight = 1)
add_edge!(g, STR2 => tan_pop2, weight = 1)
add_edge!(g, STR1 => STR2, weight = 1, t_event = 2*time_block_dur)
add_edge!(g, STR2 => STR1, weight = 1, t_event = 2*time_block_dur)
add_edge!(g, STR1 => AS)
add_edge!(g, STR2 => AS)
add_edge!(g, STR1 => SNcb, weight = 1) 
add_edge!(g, STR2 => SNcb, weight = 1)  

agent = Agent(g; name=model_name, t_block = time_block_dur); ## define agent
env = ClassificationEnvironment(stim, N_trials; name=:env, namespace=model_name)

fig = Figure(title="Adjacency matrix", size = (1600, 800))

adjacency(fig[1,1], agent)

# run the whole experiment with N_trials number of trials
t=run_experiment!(agent, env; t_warmup=200.0, alg=Vern7(), verbose=true)

#t contains the outcomes of the experiment: 
#  trials: trial number
#  correct: whether the response was correct or not
#  action: what was the responce choice, choice 1 (left saccade) or choice 2 (right saccade)
adjacency(fig[1,2], agent)

fig
