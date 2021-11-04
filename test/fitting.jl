using Neuroblox, LinearAlgebra, OrdinaryDiffEq, GalacticOptim, Optim, ForwardDiff, Test

# Create Regions
@named Str = NeuralMass(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = NeuralMass(τ=0.04, H=20, λ=400, r=0.1)
@named STN = NeuralMass(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = NeuralMass(τ=0.014, H=20, λ=400, r=0.1)
@named Th  = NeuralMass(τ=0.002, H=10, λ=20, r=5)
@named EI  = NeuralMass(τ=0.01, H=20, λ=5, r=5)
@named PY  = NeuralMass(τ=0.001, H=20, λ=5, r=0.15)
@named II  = NeuralMass(τ=2.0, H=60, λ=5, r=5)

# Connect Regions
sys = [Str, GPe, STN, GPi, Th, EI, PY, II]
@named CBGTC_Circuit = Connections(sys=sys)

sim_dur = 10.0 # Simulate for 10 Seconds
prob = ODAEProblem(structural_simplify(CBGTC_Circuit), [], (0.0, sim_dur), [])

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
