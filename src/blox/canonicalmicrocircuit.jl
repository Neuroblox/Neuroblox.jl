# Canonical micro-circuit model
# two ways to design it: either add function to edges or to blox.

# some thoughts on design:
# - include measurement models into blox. Or at least define which variables will be measured (which would be the input to the measurement model). 
#   Differ that from connector, since that is between things.
# - 

@parameters t
D = Differential(t)

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct jansen_rit_spm12 <: Blox
    τ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit_spm12(;name, τ=0.0, r=2.0/3.0)
        params = @parameters τ=τ
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, r, sigmoid(odesys.x, r), odesys)
    end
end

mutable struct cmc <: Blox
    τ::Vector{Num}
    r::Vector{Num}
    connector::Symbolics.Arr{Num}
    bloxinput::Symbolics.Arr{Num}
    odesystem::ODESystem
    function cmc(;name, τ=[0.002, 0.002, 0.016, 0.028], r=[2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0])
        @variables jcn(t)[1:4], x(t)[1:4]

        @named ss = jansen_rit_spm12(τ=τ[1], r=r[1])  # spiny stellate
        @named sp = jansen_rit_spm12(τ=τ[2], r=r[2])  # superficial pyramidal
        @named ii = jansen_rit_spm12(τ=τ[3], r=r[3])  # inhibitory interneurons granular layer
        @named dp = jansen_rit_spm12(τ=τ[4], r=r[4])  # deep pyramidal

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => ss, :name => name))
        add_vertex!(g, Dict(:blox => sp, :name => name))
        add_vertex!(g, Dict(:blox => ii, :name => name))
        add_vertex!(g, Dict(:blox => dp, :name => name))

        add_edge!(g, 1, 1, :weight, -800.0)
        add_edge!(g, 2, 1, :weight, -800.0)
        add_edge!(g, 3, 1, :weight, -800.0)
        add_edge!(g, 1, 2, :weight,  800.0)
        add_edge!(g, 2, 2, :weight, -800.0)
        add_edge!(g, 1, 3, :weight,  800.0)
        add_edge!(g, 3, 3, :weight, -800.0)
        add_edge!(g, 4, 3, :weight,  400.0)
        add_edge!(g, 3, 4, :weight, -400.0)
        add_edge!(g, 4, 4, :weight, -200.0)

        @named odecmc = ODEfromGraphdirect(g, jcn)
        # conn = [ss.connector, sp.connector, ii.connector, dp.connector]
        # the following lines are incorrect because jcn's are already assigned in ODEfromGraphdirect.
        eqs = [
            x[1] ~ ss.connector
            x[2] ~ sp.connector
            x[3] ~ ii.connector
            x[4] ~ dp.connector
        ]
        odesys = extend(ODESystem(eqs, name=:connected), odecmc, name=name)
        new(τ, r, odesys.x, odesys.jcn, odesys)
    end
end

@named foo = cmc()
equations(foo.odesystem)
states(foo.odesystem)


using ModelingToolkit

@variables t
D = Differential(t)

function linearneuralmass(;name)
    states = @variables x(t) jcn(t)
    eqs = D(x) ~ jcn
    odesys = ODESystem(eqs, t, states, []; name=name)
    return odesys, odesys.x
end

function complexblox(;name)
    @named l1 = linearneuralmass()
    @named l2 = linearneuralmass()
    @variables jcn(t)[1:2]
    @variables y(t)[1:2]

    eqs = [l1[1].jcn ~ l2[1].x + jcn[1], l2[1].jcn ~ jcn[2],
           y[1] ~ l1[2], y[2] ~ l2[2]]

    odes = ODESystem(eqs, systems=[l1[1], l2[1]], name=name)
    return odes, odes.y
end

@named com1 = complexblox()
@named com2 = complexblox()

eqs = [com1[1].jcn[1] ~ com2[2][1], com1[1].jcn[2] ~ com2[2][2], com2[1].jcn[1] ~ 0, com2[1].jcn[2] ~ 0]
@named foo = ODESystem(eqs, systems=[com1[1], com2[1]])
structural_simplify(foo)