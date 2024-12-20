# Canonical micro-circuit model

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct JansenRitSPM12 <: NeuralMassBlox
    params
    system
    namespace
    function JansenRitSPM12(;name, namespace=nothing, τ=1.0, r=2.0/3.0)
        p = paramscoping(τ=τ, r=r)
        τ, r = p

        sts    = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t) [input=true]
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]

        sys = System(eqs, t, name=name)
        new(p, sys, namespace)
    end
end

mutable struct CanonicalMicroCircuitBlox <: CompositeBlox
    namespace
    parts
    system
    connector

    function CanonicalMicroCircuitBlox(;name, namespace=nothing, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @named ss = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        sblox_parts = vcat(ss, sp, ii, dp)

        add_edge!(g, ss => ss; :weight => -800.0)
        add_edge!(g, sp => ss; :weight => -800.0)
        add_edge!(g, ii => ss; :weight => -1600.0)
        add_edge!(g, ss => sp; :weight =>  800.0)
        add_edge!(g, sp => sp; :weight => -800.0)
        add_edge!(g, ss => ii; :weight =>  800.0)
        add_edge!(g, ii => ii; :weight => -800.0)
        add_edge!(g, dp => ii; :weight =>  400.0)
        add_edge!(g, ii => dp; :weight => -400.0)
        add_edge!(g, dp => dp; :weight => -200.0)

        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(sblox_parts; name)

        new(namespace, sblox_parts, sys, bc)
    end
end
