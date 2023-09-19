using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random


"""
Tests formerly in components.jl
"""

"""
wilson_cowan test

Test for Wilson-Cowan model
"""
sim_dur = 10.0 #unsure where the sim_dur came from originally because it wasn't part of the test, so this is arbitrary
@named WC = WilsonCowanBlox()
sys = [WC.odesystem]
eqs = [sys[1].jcn ~ 0.0, sys[1].P ~ 0.0]
@named WC_sys = ODESystem(eqs,systems=sys)
WC_sys_s = structural_simplify(WC_sys)
prob = ODEProblem(WC_sys_s, [], (0,sim_dur), [])
sol = solve(prob,AutoVern7(Rodas4()),saveat=0.01)
#@test sol[1,end] ≈ 0.17513685727060388