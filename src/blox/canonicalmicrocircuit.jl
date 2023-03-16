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

        @named odecmc = ODEfromGraphdirect(g)

        # the following lines are incorrect because jcn's are already assigned in ODEfromGraphdirect.
        eqs = [
            ss.odesystem.jcn ~ jcn[1]
            sp.odesystem.jcn ~ jcn[2]
            ii.odesystem.jcn ~ jcn[3]
            dp.odesystem.jcn ~ jcn[4]
            ss.connector ~ x[1]
            sp.connector ~ x[2]
            ii.connector ~ x[3]
            dp.connector ~ x[4]
        ]
        odesys = extend(ODESystem(eqs, name=:connected), odecmc, name=name)
        new(τ, r, odesys.x, odesys.jcn, odesys)
    end
end