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
mutable struct jansen_rit_spm12
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

mutable struct cmc
    τ::Vector{Num}
    r::Vector{Num}
    odesystem::ODESystem
    lngraph::MetaDiGraph
    function cmc(;name, τ=[0.002, 0.002, 0.016, 0.028], r=[2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0])
        ss = jansen_rit_spm12(τ=τ[1], r=r[1], name=Symbol(String(name)*"_ss"))  # spiny stellate
        sp = jansen_rit_spm12(τ=τ[2], r=r[2], name=Symbol(String(name)*"_sp"))  # superficial pyramidal
        ii = jansen_rit_spm12(τ=τ[3], r=r[3], name=Symbol(String(name)*"_ii"))  # inhibitory interneurons granular layer
        dp = jansen_rit_spm12(τ=τ[4], r=r[4], name=Symbol(String(name)*"_dp"))  # deep pyramidal

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

        odesys = ODEfromGraph(g, name=name)
        new(τ, r, odesys, g)
    end
end