# Canonical micro-circuit model
# two ways to design it: either add function to edges or to blox.

# some thoughts on design:
# - include measurement models into blox. Or at least define which variables will be measured (which would be the input to the measurement model). 
#   Differ that from connector, since that is between things.
# - 

@parameters t
D = Differential(t)

# define a sigmoid function
sigmoid(x::Real, r::Real) = one(x) / (one(x) + exp(-r*x))

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct jansen_rit_spm12 <: AbstractComponent
    τ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit_spm12(;name, τ=1.0, r=2.0/3.0)
        params = @parameters τ=τ
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, r, sigmoid(odesys.x, r), odesys)
    end
end

mutable struct CanonicalMicroCircuitBlox <: SuperBlox
    τ_ss::Num
    τ_sp::Num
    τ_ii::Num
    τ_dp::Num
    r_ss::Num
    r_sp::Num
    r_ii::Num
    r_dp::Num
    connector::Symbolics.Arr{Num}
    noDetail::Vector{Num}
    detail::Vector{Num}
    bloxinput::Symbolics.Arr{Num}
    odesystem::ODESystem
    function CanonicalMicroCircuitBlox(;name, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @variables jcn(t)[1:4], x(t)[1:4]

        @named ss = jansen_rit_spm12(τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = jansen_rit_spm12(τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = jansen_rit_spm12(τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = jansen_rit_spm12(τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => ss, :name => name, :jcn => jcn[1]))
        add_vertex!(g, Dict(:blox => sp, :name => name, :jcn => jcn[2]))
        add_vertex!(g, Dict(:blox => ii, :name => name, :jcn => jcn[3]))
        add_vertex!(g, Dict(:blox => dp, :name => name, :jcn => jcn[4]))

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

        @named odecmc = ODEfromGraph(g)
        eqs = [
            x[1] ~ ss.connector
            x[2] ~ sp.connector
            x[3] ~ ii.connector
            x[4] ~ dp.connector
        ]
        odesys = extend(ODESystem(eqs, t, name=:connected), odecmc, name=name)
        new(τ_ss, τ_sp, τ_ii, τ_dp, r_ss, r_sp, r_ii, r_dp, odesys.x, [odesys.ss.x,odesys.sp.x,odesys.ii.x,odesys.dp.x], [odesys.ss.x,odesys.sp.x,odesys.ii.x,odesys.dp.x], odesys.jcn, odesys)
    end
end
