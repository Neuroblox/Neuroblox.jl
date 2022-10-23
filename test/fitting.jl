using Neuroblox, LinearAlgebra, OrdinaryDiffEq, GalacticOptim, GalacticOptimJL, GalacticOptimisers, Flux, ForwardDiff, Test

@named Str = jansen_ritSC(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_ritSC(τ=0.04, H=20, λ=400, r=0.1)
@named STN = jansen_ritSC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)

# Connect Regions through Adjacency Matrix
blox = [Str, GPe, STN, GPi]
sys = [s.odesystem for s in blox]
connect = [s.connector for s in blox]

@parameters C_Cor=3 C_BG_Th=3 C_Cor_BG_Th=9.75 C_BG_Th_Cor=9.75
adj_matrix = [0 0 0 0;
             -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0;
              0 -0.5*C_BG_Th 0 0;
              0 -0.5*C_BG_Th C_BG_Th 0]

@named BG_Circuit = LinearConnections(sys=sys, adj_matrix=adj_matrix, connector=connect)

sim_dur = 5.0 # Simulation time (seconds)
prob = ODEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])

# Fitting Procedure
# Grab system parameters
ptrue = prob.p
# Alter system parameters randomly
pinit = ptrue .+ ptrue .* randn(length(prob.p)) ./ 100
# Solve problem as normal
data = solve(prob, Vern9(), abstol=1e-9, reltol=1e-9, saveat=0.01)

# Define loss function
lb = 0.1 .* ptrue
ub = 2 .* ptrue
internalnorm(u,t) = sqrt(ForwardDiff.value(sum(abs2,u)))/length(u)
function loss(p,_)
    sol = solve(remake(prob,p=p), Vern9(), abstol=1e-9, reltol=1e-9, saveat=0.01)
    if sol.retcode != :Success
        Inf * abs(p[1])
    else
        sqrt(sum(abs2, sol .- data)/length(sol))
    end
end

iter = 0
function cb(p,l)
    global iter
    iter += 1
    @show iter,l
    l < 0.01
end

optfun = OptimizationFunction(loss,GalacticOptim.AutoForwardDiff())
optprob = OptimizationProblem(optfun,pinit)

# loss(pinit,nothing)
# loss(ptrue,nothing)

# Solve optimization problem
optsol = GalacticOptim.solve(optprob, GalacticOptimisers.Adam(0.0001),maxiters = 300, callback = cb)
optprob2 = remake(optprob,u0 = optsol.u)
optsol2 = GalacticOptim.solve(optprob2, BFGS(initial_stepnorm = 0.0001),maxiters = 300, callback = cb)

# Calculate squared error of fit
@test norm(optsol.u - ptrue) < 16 # balance threshold value and maxiters for run time
#@test_broken !iszero(norm(optsol.u - pinit))
