using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

"""
Tests formerly in components.jl
"""

"""
Larter-Breakspear model test
"""
@named lb = LarterBreakspearBlox()
sys = [lb.odesystem]
eqs = [sys[1].jcn ~ 0]
@named lb_connect = ODESystem(eqs,systems=sys)
lb_simpl = structural_simplify(lb_connect)

@test length(states(lb_simpl)) == 3

prob = ODEProblem(lb_simpl,[0.5,0.5,0.5],(0,10.0),[])
sol = solve(prob,Tsit5())

@test sol[1,10] ≈ -0.6246710908910991