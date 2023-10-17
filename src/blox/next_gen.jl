using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random


@variables t
D = Differential(t)

"""
New blocks
"""


"""
Original next-gen NMM from MPR 2015
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
Coupled next-gen NMM from MPR 2015
Note the parameters aren't given so they're assumed from the ones above
"""

struct MPRMassEI
    params
    output
    jcn
    odesystem
    namespace

    function MPRMassEI(; name, J_ee=15, J_ie=15, J_ei=15, J_ii=15, η̄ₑ=-5, η̄ᵢ=-5, Δₑ=1, Δᵢ=1)
        p = progress_scope(@parameters J_ee=J_ee J_ie=J_ie J_ei=J_ei J_ii=J_ii η̄ₑ=η̄ₑ η̄ᵢ=η̄ᵢ Δₑ=Δₑ Δᵢ=Δᵢ)
        J_ee, J_ie, J_ei, J_ii, η̄ₑ, η̄ᵢ, Δₑ, Δᵢ = p
        sts = @variables rₑ(t)=0.0 vₑ(t)=0.0 rᵢ(t)=0.0 vᵢ(t)=0.0 jcnₑ(t)=6.0 jcnᵢ(t)=0.0
        eqs = [D(rₑ) ~ (Δₑ/π) + (2*rₑ*vₑ),
                D(vₑ) ~ vₑ^2 + η̄ₑ + (J_ee*rₑ) - (π*rₑ)^2 - (J_ie*rᵢ) + jcnₑ,
                D(rᵢ) ~ (Δᵢ/π) + (2*rᵢ*vᵢ),
                D(vᵢ) ~ vᵢ^2 + η̄ᵢ + (J_ei*rₑ) - (π*rᵢ)^2 - (J_ii*rᵢ) + jcnᵢ,
                D(jcnₑ) ~ 0,
                D(jcnᵢ) ~ 0]
        sys = System(eqs; name=name)
        new(p, sts[2], sts[5], sys, sts)
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

@named hmm = MPRMassEI()

g = MetaDiGraph()
add_blox!(g, hmm)
@named final_system = system_from_graph(g)
final_system = structural_simplify(final_system)
sim_dur = 60.0
prob = ODEProblem(final_system, [], (0.0, sim_dur))
sol = solve(prob, AutoVern7(Rodas4()); saveat=0.001)