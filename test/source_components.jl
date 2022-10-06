using Neuroblox, OrdinaryDiffEq, Statistics

"""
CosineBlox Source Test
"""
# Compare CosineBlox (_nb) to Cosine from MTK Standard Library
@parameters t
@named int = Integrator()
# CosineBlox
@named src_nb   = cosine_source(f=1, a=2, phi=0, offset=1, tstart=2)
@named iosys_nb = ODESystem([connect(src_nb.odesystem.output, int.input)], t, systems = [int, src_nb.odesystem])
sys_nb = structural_simplify(iosys_nb)
# Cosine MTK
@named src     = Cosine(frequency=1, amplitude=2, phase=0, offset=1, start_time=2)
@named iosys   = ODESystem([connect(src.output, int.input)], t, systems = [int, src])
sys = structural_simplify(iosys)
# Compare Results
prob_nb = ODEProblem(sys_nb, Pair[int.x => 0.0], (0.0, 10.0))
prob    = ODEProblem(sys,    Pair[int.x => 0.0], (0.0, 10.0))
sol_nb  = solve(prob_nb, Rodas4())
sol     = solve(prob,    Rodas4())
@test mean(sol_nb[1,:]) ≈ mean(sol[1,:])