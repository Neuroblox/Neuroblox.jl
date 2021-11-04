using Neuroblox, LinearAlgebra, OrdinaryDiffEq, GalacticOptim, Optim, ForwardDiff, Test

# Create Regions
@named GPe = NeuralMass_aTan(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
@named STN = NeuralMass_aTan(ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)

# Connect Regions through Adjacency Matrix
sys = [GPe, STN]
@parameters g_GPe_STN=-1.0 g_STN_GPe=1.0
adj_matrix = [sys[1].x g_STN_GPe*sys[2].x; 
             g_GPe_STN*sys[1].x sys[2].x]

@named BG_Circuit = Connections(sys=sys, adj_matrix=adj_matrix)
sim_dur = 10.0 # Simulate for 10 Seconds
prob = ODAEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])
sol = solve(prob, Tsit5())

ptrue = prob.p
pinit = ptrue .+ ptrue .* randn(length(prob.p)) ./ 1000
data = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.1)

lb = 0.1 .* ptrue
ub = 2 .* ptrue

internalnorm(u,t) = sqrt(ForwardDiff.value(sum(abs2,u)))/length(u)

function loss(p,_)
    sol = solve(remake(prob,p=p), Tsit5(), abstol=1e-12, reltol=1e-12, saveat=0.1)
    if sol.retcode != :Success
        Inf * p[1]
    else
        sum(abs2, sol .- data)
    end
end
optfun = OptimizationFunction(loss,GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(optfun,pinit,lb=lb,ub=ub)
optsol = solve(optprob,GradientDescent(),maxiters = 10000)

@test_broken !iszero(norm(optsol.u - pinit))
#@test norm(optsol.u - ptrue) < 1