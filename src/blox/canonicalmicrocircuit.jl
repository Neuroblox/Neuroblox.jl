# Canonical micro-circuit model

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct JansenRitSPM12 <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function JansenRitSPM12(;name, namespace=nothing, τ=1.0, r=2.0/3.0)
        p = paramscoping(τ=τ, r=r)
        τ, r = p

        sts    = @variables x(t)=1.0 [output=true] y(t)=1.0 jcn(t)=0.0 [input=true]
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]

        sys = System(eqs, t, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end

mutable struct CanonicalMicroCircuitBlox <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    function CanonicalMicroCircuitBlox(;name, namespace=nothing, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @named ss = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        sblox_parts = vcat(ss, sp, ii, dp)
        add_blox!.(Ref(g), sblox_parts)

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

        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(sblox_parts; name)

        new(namespace, sblox_parts, sys, bc)
    end
end
