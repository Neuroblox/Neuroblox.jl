# Canonical micro-circuit model
# two ways to design it: either add function to edges or to blox.

# some thoughts on design:
# - include measurement models into blox. Or at least define which variables will be measured (which would be the input to the measurement model). 
#   Differ that from connector, since that is between things.
# - 

@parameters t
D = Differential(t)

"""
Jansen-Rit model blox for canonical micro circuit
"""
mutable struct jansen_rit4cmc <: JansenRitCBlox
    τ::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit4cmc(;name, τ=0.0)
        params = @parameters τ=τ
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, sigmoid(odesys.x, 2/3), odesys)
    end
end

mutable struct cmc
    τ::Vector{Float64}
    connector::Num
    odesystem::ODESystem
    graph::LinearNeuroGraph
    function cmc(;name, τ=[0.002, 0.002, 0.016, 0.028])
        @named ss = jansen_rit4cmc(τ=τ[1])  # spiny stellate
        @named sp = jansen_rit4cmc(τ=τ[2])  # superficial pyramidal
        @named ii = jansen_rit4cmc(τ=τ[3])  # inhibitory interneurons granular layer
        @named dp = jansen_rit4cmc(τ=τ[4])  # deep pyramidal

        g = LinearNeuroGraph(MetaDiGraph())
        add_blox!(g, ss)
        add_blox!(g, sp)
        add_blox!(g, ii)
        add_blox!(g, dp)

        add_edge!(g, 1, 1, :weight, -800.0)
        add_edge!(g, 2, 1, :weight, -800.0)   # from ii to ss
        add_edge!(g, 3, 1, :weight, -800.0)
        add_edge!(g, 1, 2, :weight,  800.0)
        add_edge!(g, 2, 2, :weight, -800.0)
        add_edge!(g, 1, 3, :weight,  800.0)
        add_edge!(g, 3, 3, :weight, -800.0)
        add_edge!(g, 4, 3, :weight,  400.0)
        add_edge!(g, 3, 4, :weight, -400.0)
        add_edge!(g, 4, 4, :weight, -200.0)

        odesys = ODEfromGraph(g=g, name=name)
        new(τ, odesys.sp.x + odesys.dp.x, odesys, g)
    end
end