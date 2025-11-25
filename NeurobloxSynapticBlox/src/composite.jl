mutable struct WinnerTakeAll <: AbstractComposite
    name::Symbol
    namespace::Union{Symbol, Nothing}
    inhi::HHInhi
    excis::Vector{HHExci}
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function WinnerTakeAll(;name,
                           namespace=nothing,
                           N_exci = 5,
                           E_syn_exci=0.0,
                           E_syn_inhib=-70,
                           G_syn_exci=3.0,
                           G_syn_inhib=3.0,
                           I_bg=zeros(N_exci),
                           # phase=0.0,
                           τ_exci=5,
                           τ_inhib=70)
        
        inhi = HHInhi(name = :inh, namespace = namespaced_name(namespace, name), E_syn = E_syn_inhib, G_syn = G_syn_inhib, τ = τ_inhib)
        
        excis = map(1:N_exci) do i
            HHExci(
                name = Symbol("exci$i"),
                namespace = namespaced_name(namespace, name), 
                I_bg = (I_bg isa AbstractArray) ? I_bg[i] : I_bg*rand(), # behave differently if I_bg is array
                E_syn = E_syn_exci,
                G_syn = G_syn_exci,
                τ = τ_exci
            )
        end
        g = GraphSystem()
        for excii ∈ excis
            system_wiring_rule!(g, inhi, excii; weight=1.0)
            system_wiring_rule!(g, excii, inhi; weight=1.0)
        end
        new(name, namespace, inhi, excis, [inhi; excis], g, nothing, nothing)
    end
end

mutable struct Cortical <: AbstractComposite
    name::Symbol
    namespace::Union{Symbol, Nothing}
    wtas
    n_ff_inh
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function Cortical(;name,
                      namespace=nothing,
                      N_wta=20,
                      N_exci=5,
                      E_syn_exci=0.0,
                      E_syn_inhib=-70,
                      G_syn_exci=3.0,
                      G_syn_inhib=4.0,
                      G_syn_ff_inhib=3.5,
                      I_bg_ar=0,
                      τ_exci=5,
                      τ_inhib=70,
                      kwargs...)

        g = GraphSystem()
        n_ff_inh = HHInhi(name=:ff_inf, namespace=namespaced_name(namespace, name), E_syn=E_syn_inhib, G_syn=G_syn_ff_inhib, τ=τ_inhib)
        system_wiring_rule!(g, n_ff_inh)
        wtas = map(1:N_wta) do i
            if I_bg_ar isa AbstractArray
                I_bg = I_bg_ar[i]
            else
                I_bg = I_bg_ar
            end
            WinnerTakeAll(; name=Symbol("wta$i"),
                          namespace = namespaced_name(namespace, name),
                          N_exci,
                          E_syn_exci,
                          E_syn_inhib,
                          G_syn_exci,
                          G_syn_inhib,
                          I_bg = I_bg,
                          τ_exci,
                          τ_inhib)
        end
        # users can supply a matrix of connection matrices.
        # connection_matrices[i,j][k, l] determines if neuron k from l-flic i is connected to
        # neuron l from l-flic j.
        connection_matrices = get(kwargs, :connection_matrices) do
            map(Iterators.product(1:N_wta, 1:N_wta)) do (i, j)
                get_connection_matrix(kwargs,
                                      Symbol("wta$i"), Symbol("wta$j"),
                                      N_exci, N_exci)
            end
        end
        for i ∈ 1:N_wta
            system_wiring_rule!(g, wtas[i])
            for j ∈ 1:N_wta
                if j != i
                    kwargs_ij = merge(kwargs, Dict(:connection_matrix => connection_matrices[i, j]))
                    system_wiring_rule!(g, wtas[i], wtas[j]; kwargs_ij...)
                end
            end
            # connect the inhibitory neuron to the i-th wta
            system_wiring_rule!(g, n_ff_inh, wtas[i]; weight=1.0)
        end
        new(name, namespace, wtas, n_ff_inh, [wtas; n_ff_inh], g, nothing, nothing)
    end
end

mutable struct Striatum <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHInhi}
    matrisome::Matrisome
    striosome::Striosome
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function Striatum(; name,
                      namespace=nothing,
                      N_inhib = 25,
                      E_syn_inhib=-70,
                      G_syn_inhib=1.2,
                      I_bg=zeros(N_inhib),
                      τ_inhib=70
                      )

        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            HHInhi(
                name = Symbol("inh$i"),
                namespace = inner_namespace,
                E_syn = E_syn_inhib, 
                G_syn = G_syn_inhib, 
                τ = τ_inhib,
                I_bg = I_bg[i],
            ) 
        end
        matrisome = Matrisome(; name=:matrisome, namespace=inner_namespace)
        striosome = Striosome(; name=:striosome, namespace=inner_namespace)
        
        g = GraphSystem()

        for n ∈ inhibs
            system_wiring_rule!(g, n)
        end
        system_wiring_rule!(g, matrisome)
        system_wiring_rule!(g, striosome)

        new(name, namespace, inhibs, matrisome, striosome, [inhibs; matrisome; striosome],
            g, nothing, nothing)
    end
end


function NeurobloxBase.connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, get_matrisome(str1), get_matrisome(str2))
end

NeurobloxPharma.get_striosome(str::Striatum) = str.striosome
NeurobloxPharma.get_matrisome(str::Striatum) = str.matrisome


mutable struct GPi <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHInhi}
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function GPi(; name,
                 namespace=nothing,
                 N_inhib = 25,
                 E_syn_inhib =-70,
                 G_syn_inhib=8,
                 I_bg=4*ones(N_inhib),
                 τ_inhib=70)

        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            inhib = HHInhi(; name=Symbol(:inh, i),
                           namespace = inner_namespace,
                           E_syn = E_syn_inhib,
                           G_syn = G_syn_inhib,
                           τ = τ_inhib,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, namespace, inhibs, inhibs, graph, nothing, nothing)
    end
end

mutable struct GPe <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHInhi}
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function GPe(;name,
                 namespace=nothing,
                 N_inhib=15,
                 E_syn_inhib=-70,
                 G_syn_inhib=3,
                 I_bg=2*ones(N_inhib),
                 τ_inhib=70)
        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            inhib = HHInhi(; name=Symbol(:inh, i),
                           namespace=inner_namespace,
                           E_syn = E_syn_inhib,
                           G_syn = G_syn_inhib,
                           τ = τ_inhib,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, namespace, inhibs, inhibs, graph, nothing, nothing)
    end
end

mutable struct Thalamus <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    excis::Vector{HHExci}
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function Thalamus(;name,
                      namespace=nothing,
                      N_exci = 25,
                      E_syn_exci=0,
                      G_syn_exci=3,
                      I_bg=3*ones(N_exci),
                      τ_exci=5,
                      density=0.0,
                      weight=1.0,
                      connection_matrix=subcortical_connection_matrix(density, N_exci, weight))
        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        excis = map(1:N_exci) do i
            exci = HHExci(; name=Symbol(:exci, i),
                          namespace = inner_namespace,
                          E_syn = E_syn_exci,
                          G_syn = G_syn_exci,
                          τ = τ_exci,
                          I_bg = I_bg[i])
            system_wiring_rule!(graph, exci)
            exci
        end
        for i ∈ 1:N_exci
            for j ∈ 1:N_exci
                cij = connection_matrix[i,j]
                if !iszero(cij)
                    system_wiring_rule!(graph, excis[i], excis[j], weight=cij)
                end
            end
        end
        new(name, namespace, excis, excis, graph, nothing, nothing)
    end
end

mutable struct STN <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    excis::Vector{HHExci}
    parts::Vector
    graph::GraphSystem
    system::Union{Nothing, ODESystem}
    connector::Union{Nothing, Connector, Vector{Connector}}
    function STN(;name,
                 namespace=nothing,
                 N_exci = 25,
                 E_syn_exci=0,
                 G_syn_exci=3,
                 I_bg=3*ones(N_exci),
                 τ_exci=5)
        graph = GraphSystem()
        excis = map(1:N_exci) do i
            exci = HHExci(; name=Symbol(:exci, i),
                          namespace=namespaced_name(namespace, name),
                          E_syn = E_syn_exci,
                          G_syn = G_syn_exci,
                          τ = τ_exci,
                          I_bg = I_bg[i])
            system_wiring_rule!(graph, exci)
            exci
        end
        new(name, namespace, excis, excis, graph, nothing, nothing)
    end
end




function connector_and_system((; name, namespace, graph, parts)::Union{WinnerTakeAll,
                                                                       Cortical,
                                                                       Striatum,
                                                                       GPi,
                                                                       GPe,
                                                                       Thalamus,
                                                                       STN
                                                                       })
    g = to_metadigraph(graph)
    bc = connectors_from_graph(g)
    if bc isa Vector && !isempty(bc)
        bc = reduce(merge!, bc; )
    end
    
    system = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)
    
    (; connector=bc, system)
end


function Base.getproperty(x::Union{WinnerTakeAll, Cortical, Striatum, GPi, GPe, Thalamus, STN
                                   }, s::Symbol)
    if s == :system || s == :connector
        if isnothing(getfield(x, s))
            (; connector, system) = connector_and_system(x)
            x.connector = connector
            x.system = system
        end
    end
    getfield(x, s)
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem,
                                           blox::Union{WinnerTakeAll,
                                                       Cortical,
                                                       Striatum,
                                                       GPi,
                                                       GPe,
                                                       Thalamus,
                                                       STN,
                                                       }; kwargs...)
    merge!(g, blox.graph)
end
