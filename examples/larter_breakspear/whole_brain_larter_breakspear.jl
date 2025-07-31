# This is a tutorial for how to run a whole-brain simulation using the Larter-Breakspear model.
# This approach is the same as that taken in Antal et al. (2023) and van Nieuwenhuizen et al. (2023), with the exception that a different DTI dataset is used.
# The data used in those papers is available upon request from the authors where it was validated in Endo et al. (2020).
# As we'd like to make this tutorial as accessible as possible, we'll use a different dataset that is publicly available.
# The dataset used here is from Rosen and Halgren (2021) and is available at https://zenodo.org/records/10150880.

# References:
# 1. Antal et al. (2023). DOI: 10.48550/arXiv.2303.13746.
# 2. van Nieuwenhuizen et al. (2023). DOI: 10.1101/2023.05.10.540257.
# 3. Endo et al. (2020). DOI: 10.3389/fncom.2019.00091.
# 4. Rosen and Halgren (2021). DOI: 10.1523/ENEURO.0416-20.2020.

using Neuroblox # Core functionality
using CSV, MAT, DataFrames # Import/export data functions
using MetaGraphs # Set up graph of systems
using DifferentialEquations # Needed for solver TODO: make a new simulate that can handle system_from_graph
using StatsBase # Needed for rescaling the DTI matrix
using Random # Needed to seed the random generator
#using Plots # Only uncomment if you actually want to do plotting, otherwise save yourself the overhead

# A note on the data: this is a 360-region parcellation of the brain. For this example, we'll extract and rescale the left hemisphere default mode network.
# If you want a true rescaling method, you'd need to do some additional corrections (e.g., volume correction) for the number of streamlines computed.
# For this tutorial though, the rescaling is approximate to just give us a working example.
# See Endo et al. for more details on how to do a better DTI rescaling.

# Load the data
# As mentioned above, this data is from Rosen and Halgren (2021). You can download the original at https://zenodo.org/records/10150880.
# For convenience, we have included the average connectivity matrix (log scaled probability from streamline counts) available from that link within this repository.
data = matread("averageConnectivity_Fpt.mat")

# Extract the data
adj = data["Fpt"]
adj[findall(isnan, adj)] .= minimum((filter(!isnan,adj))) # Replace NaNs with 0s - needs to be minimum for rescaling in next step
adj = StatsBase.transform(StatsBase.fit(UnitRangeTransform, adj, dims=2), adj) # Equivalent of Matlab rescale function. Resamples to unit range.

# For the purpose of this simulation, we want something that will run relatively quickly.
# In this parcellation, there are 40 regions per hemisphere in the default mode network, so we'll extract the left hemisphere DMN.
# Original indices with networks are listed in allTables.xlsx at the same download link from the Rosen and Halgren (2021) dataset listed above.
left_indices = [30, 31, 32, 33, 34, 35, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76, 87, 88, 90, 93, 94, 118, 119, 120, 126, 130, 131, 132, 134, 150, 151, 155, 161, 162, 164, 165, 176, 177]
adj = adj[left_indices, left_indices]
adj[adj .< 0.75] .= 0 # Threshold the connectivity matrix to make it sparser
adj = adj ./ 4 # Rescale so regions aren't overly connected

# Extract the names of the regions
names = data["parcelIDs"]
names = names[left_indices]

# Plot the connectivity matrix
# This is optional but gives you a sense of what the overall connectivity looks like.
# If you want to do plotting remember to uncomment the Plots import above.
#heatmap(adj)

# Number of regions in this particular dataset
N = 40

# Set the random seed for reproducibility. Because the DTI is scaled in an ad-hoc way it's not guaranteed to produce interpretable results for all random values.
# Feel free to change this, but do be cautious in examining results.
rng = MersenneTwister(42) 

# Create list of all blocks
blocks = Vector{LarterBreakspear}(undef, N)

for i in 1:N
    # Since we're creating an ODESystem inside of the blox, we need to use a symbolic name
    # Why the extra noise in connectivity? The DTI scaling is arbitrary in this demo, so adding stochasticity to this parameter helps things from just immediately synchronizing.
    blocks[i] = LarterBreakspear(name=Symbol(names[i]))
end

# Create a graph using the blocks and the DTI defined adjacency matrix
g = MetaDiGraph()
add_blox!.(Ref(g), blocks)
create_adjacency_edges!(g, adj)

# Create the ODE system
# This may take a minute, as it is compiling the whole system
@named sys = system_from_graph(g)
sys = structural_simplify(sys)

# Simulate for 100ms
sim_dur = 1e3

# Create random initial conditions because uniform initial conditions are no bueno. There are 3 states per node.
v₀ = 0.9*rand(N) .- 0.6
z₀ = 0.9*rand(N) .- 0.9
w₀ = 0.4*rand(N) .+ 0.11
u₀ = [v₀ z₀ w₀]'[:] # Trick to interleave vectors based on column-major ordering

# Create the ODEProblem to run all the final system of equations
prob = ODEProblem(sys, u₀, (0.0, sim_dur), [])

# Run the simulation and save every 2ms
@time sol = solve(prob, AutoVern7(Rodas4()), saveat=2)

# Visualizing the results
# Again, to use any of these commands be sure to uncomment the Plots import above.
# This plot shows all of the states computed. As such, it is very messy, but useful to make sure that no out-of-bounds behavior (due to bad connectivity) occured.
# plot(sol)

# More interesting is to choose the plot from a specific region and see the results. Here, we'll plot a specific region's average voltage.
# First, confirm the region (left orbitofrontal cortex)
unknowns(sys)[64] # Should give L_OFC₊V(t)
# Next plot just this variable
#plot(sol, idxs=(64))

# Suppose we want to run a simulation with different initial conditions. We can do that by remaking the problem to avoid having to recompile the entire system.
# For example, let's say we want to run the simulation with a different initial condition for the voltage.
v₀ = 0.9*rand(N) .- 0.61
z₀ = 0.9*rand(N) .- 0.89
w₀ = 0.4*rand(N) .+ 0.1
u₀ = [v₀ z₀ w₀]'[:] # Trick to interleave vectors based on column-major ordering

# Now remake and re-solve
prob2 = remake(prob; u0=u₀)
@time sol2 = solve(prob2, AutoVern7(Rodas4()), saveat=2)

# Running a longer simulation
# Now that we've confirmed this runs, let's go ahead and do a 10min (600s) simulation.
# Takes <1min to simulate this run. If you save more frequently it'll take longer to run, but you'll have more data to work with.
# Remember to update the dt below for the BOLD signal if you don't save at the same frequency.
# If you find this is taking an *excessively* long time (i.e., more than a couple minutes), you likely happened upon a parameter set that has put you into a very stiff area.
# In that case, you can try to re-run the simulation with a different initial condition or parameter set.
# In a real study, you'd allow even these long runs to finish, but for the sake of this tutorial we'll just stop it early.
sim_dur = 6e5
prob = remake(prob2; tspan=(0.0, sim_dur))
@time sol = solve(prob, AutoVern7(Rodas4()), saveat=2)

# Instead of plotting the data, let's save it out to a CSV file for later analysis
#CSV.write("example_output.csv", DataFrame(sol))

# You should see a noticeable difference in speed compared to the first time, and notice you save the overhead of structural_simplify.

# Now let's do a simulation that creates an fMRI signal out of the neural simulation run above.
# This example will use the Balloon model with slightly adjusted parameters - see Endo et al. for details.

TR = 800 # Desired repetition time (TR) in ms
dt = 2 # Time step of the simulation in ms
bold = boldsignal_endo_balloon(sol.t/1000, sol[1:3:end, :], TR, dt) # Note this returns the signal in units of TR

# Remember that the BOLD signal takes a while to equilibrate, so drop the first 90 seconds of the signal.
omit_idx = Int(round(90/(TR/1000)))
bold = bold[:, omit_idx:end]

# Plot an example region to get a sense of what the BOLD signal looks like
# plot(bold[1, :])
