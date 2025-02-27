# ## Import the necessary packages
# This isn't production quality example code. It's just to give a sense of how we'd solve
# the challenge problem in the PING tutorial.

# Reasons for each non-Neuroblox package are given in the comments after each.
using Neuroblox 
using OrdinaryDiffEq ## to build the ODE problem and solve it, gain access to multiple solvers from this
using Distributions ## for statistical distributions 
using Random ## for random number generation
using CairoMakie ## for plotting
using Peaks

Random.seed!(42);

# Solution to challenge problem part 1 - rhythmic inhibition

g_IE = 0.5 ## Reasonable range 0.1-0.8ish seems tunable
g_EI = 0.45; ## mess with this but don't put too high - fire quickly but no doublets is the goal

I_D = 1.55 ## This should be <= I_L, and is the main one to tune
I_L = 1.75 ## Set manually based on desired maxiimimum spiking rate in unconnected network
I_I = 0.0 # Don't drive inhibitory neurons

@named E1 = PINGNeuronExci(I_ext=I_D)
@named E2 = PINGNeuronExci(I_ext=I_L) 
@named I = PINGNeuronInhib(I_ext=I_I) 

g = MetaDiGraph()

add_edge!(g, E1 => I; weight=g_EI)
add_edge!(g, E2 => I; weight=g_EI)
add_edge!(g, I => E1; weight=g_IE)
add_edge!(g, I => E2; weight=g_IE)

tspan = (0.0, 1000.0) 
@named sys = system_from_graph(g) 
prob = ODEProblem(sys, [], tspan) 
sol = solve(prob, Tsit5(), saveat=0.1); 
#plot(sol)

V_D = voltage_timeseries(E1, sol)
V_L = voltage_timeseries(E2, sol)
V_I = voltage_timeseries(I, sol)

p_D = findmaxima(V_D)
count(!iszero, p_D.heights .> 10)
p_L = findmaxima(V_L)
count(!iszero, p_L.heights .> 10)
p_I = findmaxima(V_I)
count(!iszero, p_I.heights .> 10)

# Solution to challenge problem part 2 - asynchronous inhibition

I_D = 1.55 ## This should be <= I_L, and is the main one to tune
I_L = 1.75 ## Set manually based on desired maxiimimum spiking rate in unconnected network
I_asynch = -1.3 # Asynchronous drive

@named E1 = PINGNeuronExci(I_ext=I_D+I_asynch)
@named E2 = PINGNeuronExci(I_ext=I_L+I_asynch) 

g = MetaDiGraph()
add_blox!.(Ref(g), [E1, E2])

tspan = (0.0, 1000.0) 
@named sys = system_from_graph(g) 
prob = ODEProblem(sys, [], tspan) 
sol = solve(prob, Tsit5(), saveat=0.1); 
#plot(sol)

V_D = voltage_timeseries(E1, sol)
V_L = voltage_timeseries(E2, sol)

p_D = findmaxima(V_D)
count(!iszero, p_D.heights .> 10)
p_L = findmaxima(V_L)
count(!iszero, p_L.heights .> 10)
