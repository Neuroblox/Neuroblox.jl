using Neuroblox, OrdinaryDiffEq, Test, Plots

"""
Design Test Case. 
A circuit model is used, with an explicit paramter setting the oscillation frequency range of the system in radians/second:
    ω = 4*2*π
The test will count the number of peaks in a given time window and match that to the frequency containing the most power.

"""

freq_of_interest = 4

# Create Circuit 
@named STN = NeuralMass(activation="a_tan", ω=freq_of_interest*2*π, ζ=1, k=(freq_of_interest*2*π)^2, h=5.0)
sys = [STN]
adj_matrix = [1.0]
@named BG_Circuit = LinearConnections(sys=sys, adj_matrix=adj_matrix)
# Run Simulation
sim_dur = 5.0                                                                           # Simulation time (seconds)
prob = ODAEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5(), dt = 0.001)

"""
Perform Test.
The power spectrum of the solution should have a center frequency around that set by ω in Hz, with some tolerance.
To Do: Employ a peak detection algorithm, useful for this test and signal processing. Number of peaks should match center frequency.

"""
@named power = Neuroblox.PowerSpectrum(data=sol[1,:], T=sim_dur, uniform=true, dt=0.001)

# 1) Make sure you can plot the data
Plots.plot(sol.t, sol[1,:])
Plots.plot(power[1], power[2], xlims=(0,10))

# 2) Find the local maxima
T = sim_dur
df = 1/T
find_max = findmax(power[2][1:length(power[1])-Int(ceil(freq_of_interest/df))])
index_of_maximum = find_max[2]

tol = 0.5
@test index_of_maximum*df>4-tol
@test index_of_maximum*df<4+tol
