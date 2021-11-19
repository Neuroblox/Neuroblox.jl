using Neuroblox, LinearAlgebra, OrdinaryDiffEq, ForwardDiff, Test

using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Turing
using Distributions
using MCMCChains
using StatsPlots


using Random
using Plots
using Turing: Variational

## This model works:
# Create Regions
#@named GPe = NeuralMass(activation="a_tan", ω=4*2*π, ζ=1, k=(4*2*π)^2, h=0.1)
#@named STN = NeuralMass(activation="a_tan", ω=4*2*π, ζ=1, k=(4*2*π)^2, h=1)
# Connect Regions through Adjacency Matrix
#sys = [GPe, STN]
#@parameters g_GPe_STN=1.0 g_STN_GPe=1.0
#adj_matrix = [sys[1].x g_STN_GPe*sys[2].x; 
#             -g_GPe_STN*sys[1].x sys[2].x]

## Try a more complicated circuit: It works 
#@named GPe = NeuralMass(activation="logistic", τ=0.04, H=20, λ=400, r=0.1)
#@named STN = NeuralMass(activation="logistic", τ=0.01, H=20, λ=500, r=0.1)
#@named GPi = NeuralMass(activation="logistic", τ=0.014, H=20, λ=400, r=0.1)

# Connect Regions through Adjacency Matrix
#sys = [GPe, STN, GPi]
#@parameters C_Cor=3 C_BG_Th=3 C_Cor_BG_Th=9.75 C_BG_Th_Cor=9.75
#adj_matrix = [-0.5*C_BG_Th*sys[1].x C_BG_Th*sys[2].x 0;
#              -0.5*C_BG_Th*sys[1].x 0 0
#              -0.5*C_BG_Th*sys[1].x C_BG_Th*sys[2].x 0]

# Try an even more complicated circuit: It works!
@named Str = NeuralMass(activation="logistic", τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = NeuralMass(activation="logistic", τ=0.04, H=20, λ=400, r=0.1)
@named STN = NeuralMass(activation="logistic", τ=0.01, H=20, λ=500, r=0.1)
@named GPi = NeuralMass(activation="logistic", τ=0.014, H=20, λ=400, r=0.1)

# Connect Regions through Adjacency Matrix
sys = [Str, GPe, STN, GPi]
@parameters C_Cor=3 C_BG_Th=3 C_Cor_BG_Th=9.75 C_BG_Th_Cor=9.75
adj_matrix = [0 0 0 0;
             -0.5*C_BG_Th*sys[1].x -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0;
              0 -0.5*C_BG_Th*sys[2].x 0 0;
              0 -0.5*C_BG_Th*sys[2].x C_BG_Th*sys[3].x 0]

@named BG_Circuit = Connections(sys=sys, adj_matrix=adj_matrix)

sim_dur = 5.0 # Simulation time (seconds)
prob1 = ODAEProblem(structural_simplify(BG_Circuit), [], (0.0, sim_dur), [])

# Fitting Procedure
# Grab system parameters
ptrue = prob.p
# Alter system parameters randomly
pinit = ptrue .+ ptrue .* randn(length(prob.p)) ./ 100
# Solve problem as normal
data = solve(prob1, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.001)

lb = 0.1 .* ptrue
ub = 2 .* ptrue

#prob2 = remake(prob1,p=pinit)


#data = solve(prob1, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.1)
#data2= solve(prob2, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.1)

Turing.setadbackend(:forwarddiff)

  @model function fitNeuralMass(data,prob1)
    σ ~ InverseGamma(2,3)
    
    #params : g_GPe_STN, g_STN_GPe, GPe_ω, GPe_ζ, Gpe_k, Gpe_h,  STN_ω, STN_ζ, STN_k, STN_h, 
    
    p = Vector{Real}(undef,length(ptrue))
    for i = 1:length(ptrue)
       p[i] ~ Uniform(lb[i],ub[i])
    end

    # WIP: Add new feature 
   u0 = typeof(p[1]).(prob1.u0)
   #typeof(p) <: Dual
   #typeof(p) === Vector{Any}
   #any(x->typeof(x)<: Dual,p)
   u0 = u0 

   # @named GPe2 = NeuralMass(activation="a_tan", ω=p[3], ζ=p[4], k=p[5], h=p[6])
    #@named STN2 = NeuralMass(activation="a_tan", ω=p[7], ζ=p[8], k=p[9], h=p[10])
    
    #sys2 = [GPe2, STN2]
    #params2 =  @parameters g_GPe_STN2=p[1] g_STN_GPe2=p[2]

    #adj_matrix2 = [sys2[1].x g_STN_GPe2*sys2[2].x; 
    #          -g_GPe_STN2*sys2[1].x sys2[2].x]



    #@named BG_Circuit2 = Connections(sys=sys2, adj_matrix=adj_matrix2)
     
    #sim_dur2 = 10.0 # Simulate for 10 Seconds
    #prob2 = ODAEProblem(structural_simplify(BG_Circuit2), [], (0.0, sim_dur2), [])
   
    prob = remake(prob1,u0=u0,p=p)
    predicted = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.1)
    
    #if predicted.retcode != :Success
    #  Turing.acclogp!(_varinfo, -Inf)
    #end
    
    for i = 1:length(predicted)
      data[i] ~ MvNormal(predicted[i],σ)
    end
    
  end

  model = fitNeuralMass(data,prob1)

  #chain = sample(model, NUTS(0.65), MCMCThreads(),5000, 4,init_theta = pinit)
  #plot(chain)



  advi = ADVI(10, 1000)
  q = vi(model, advi);

  samples=rand(q,10000)

  pl= plot(data,alpha=2)
  for k in 1:300
    resol = solve(remake(prob1,p=samples[2:(length(ptrue)+1),rand(1:10000)]),Vern9(), abstol=1e-12, reltol=1e-12, saveat=0.1)
    plot!(pl,resol, alpha=0.3, color = "#BBBBBB", legend = false)
end