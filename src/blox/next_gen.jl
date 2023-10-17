using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random


@variables t
D = Differential(t)

"""
New blocks
"""

struct MPRSingleMass
    params
    output
    jcn
    odesystem
    namespace

    function MPRSingleMass(; name, J=15, η̄=-5, Δ=1)
        p = progress_scope(@parameters J=J η̄=η̄ Δ=Δ)
        J, η̄, Δ = p
        sts = @variables r(t)=0.0 v(t)=-2.0 jcn(t)=3.0
        eqs = [D(r) ~ (Δ/π) + (2*r*v),
               D(v) ~ v^2 + η̄ + (J*r) - π^2*r^2 + jcn,
               D(jcn) ~ 0]
        sys = System(eqs; name=name)
        new(p, sts[2], sts[3], sys, sts)
    end
end





"""
TESTS
"""
@named hmm = MPRSingleMass()

g = MetaDiGraph()
add_blox!(g, hmm)
@named final_system = system_from_graph(g)
final_system = structural_simplify(final_system)
sim_dur = 60.0
prob = ODEProblem(final_system, [], (0.0, sim_dur))
sol = solve(prob, AutoVern7(Rodas4()); saveat=0.001)