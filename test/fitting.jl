using Neuroblox, LinearAlgebra, OrdinaryDiffEq, GalacticOptim, Optim, ForwardDiff, Test

# Create Regions
@named GPe = NeuralMass(activation="a_tan", ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = NeuralMass(activation="a_tan", ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)

# Connect Regions through Adjacency Matrix
sys = [GPe, STN]
@parameters g_GPe_STN=1.0 g_STN_GPe=1.0
adj_matrix = [sys[1].x g_STN_GPe*sys[2].x; 
             -g_GPe_STN*sys[1].x sys[2].x]
@named BG_Circuit = Connections(sys=sys, adj_matrix=adj_matrix)

sim_dur = 5.0 # Simulation time (seconds)
prob = ODAEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])

# Fitting Procedure
# Grab system parameters
ptrue = prob.p
# Alter system parameters randomly
pinit = ptrue .+ ptrue .* randn(length(prob.p)) ./ 100
# Solve problem as normal
data = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.001)

# Define loss function
lb = 0.1 .* ptrue
ub = 2 .* ptrue
internalnorm(u,t) = sqrt(ForwardDiff.value(sum(abs2,u)))/length(u)
function loss(p,_)
    sol = solve(remake(prob,p=p), Tsit5(), abstol=1e-12, reltol=1e-12, saveat=0.001)
    if sol.retcode != :Success
        Inf * abs(p[1])
    else
        sum(abs2, sol .- data) 
    end
end
optfun = OptimizationFunction(loss,GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(optfun,pinit,lb=lb,ub=ub)

# Solve optimization problem
optsol = solve(optprob, BFGS(),maxiters = 200)

# Calculate squared error of fit
@test norm(optsol.u - ptrue) < 1 # balance threshold value and maxiters for run time
#@test_broken !iszero(norm(optsol.u - pinit))
