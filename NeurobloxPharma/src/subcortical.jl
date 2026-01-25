"""
    Subcortical blox
    all subcortical blox used in cortico-striatal model are defined here

"""

# internal connectivity matrix
function subcortical_connection_matrix(density, N, weight; rng=Random.default_rng())
    connection_matrix = zeros(N, N)
    idxs = 1:N
    for i in idxs
        for j in idxs
            if !(i==j) && (rand(rng)<=density && connection_matrix[j, i] == 0)
                connection_matrix[i, j] = weight 
            end
        end
    end
    connection_matrix
end

"""
    Striatum(N_inhib = 25,
            E_syn_inhib=-70,
            G_syn_inhib=1.2,
            I_bg=zeros(N_inhib),
            τ_inhib=70.0
    )

A model of the Striatum that contains three components; [`Matrisome`](@ref), [`Striosome`](@ref) and an additional group of [`HHNeuronInhib`](@ref) neurons representing Medium Spiny Neurons (MSNs). There are no connections between these clusters, however they may receive connections from or project to other blox components in a model, depending on what other regions are part of the model and the connection rules between Striatum and these regions. 

Arguments : 
- `N_inhib` : Number of inhibitory neurons.
- `E_syn_inhib` [mV, reversal potential for GABA A receptors]
- `G_syn_inhib` [mV, GABA A receptor conductance]
- `I_bg` : [μA] Background current applied to the additional cluster of inhibitory neurons. If it is a single value then the same current is applied to all `N_inhib` inhibitory neurons. If it is a Vector then it needs to be of length `N_inhib` and each element is applied to one neuron.
- `τ_inhib` : [ms, decay time constant for GABA A receptor conductance]

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.

See also [`Matrisome`](@ref), [`Striosome`](@ref) and [`HHNeuronInhib`](@ref).
"""
struct Striatum <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHNeuronInhib}
    matrisome::Matrisome
    striosome::Striosome
    graph::GraphSystem
    function Striatum(; name,
                      namespace=nothing,
                      N_inhib = 25,
                      E_syn_inhib=-70,
                      G_syn_inhib=1.2,
                      I_bg=zeros(N_inhib),
                      τ_inhib=70.0
                      )

        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            HHNeuronInhib(
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

        new(name, namespace, inhibs, matrisome, striosome, g)
    end
end

get_striosome(str::Striatum) = str.striosome
get_matrisome(str::Striatum) = str.matrisome

"""
    GPi(; N_inhib = 25,
        E_syn_inhib =-70,
        G_syn_inhib=8,
        I_bg=4*ones(N_inhib),
        τ_inhib=70.0
)

A model of Globus Pallidus Internus (GPi). A group of [`HHNeuronInhib`](@ref) neurons, which are typically project to the inhibitory neurons in [`Striatum`](@ref).

Arguments : 
- `N_inhib` : Number of inhibitory neurons.
- `E_syn_inhib` [mV, reversal potential for GABA A receptors]
- `G_syn_inhib` [mV, GABA A receptor conductance]
- `I_bg` : [μA] Background current applied to all neurons. If it is a single value then the same current is applied to all `N_inhib` inhibitory neurons. If it is a Vector then it needs to be of length `N_inhib` and each element is applied to one neuron.
- `τ_inhib` : [ms, decay time constant for GABA A receptor conductance]

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
struct GPi <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHNeuronInhib}
    graph::GraphSystem
    function GPi(; name,
                 namespace=nothing,
                 N_inhib = 25,
                 E_syn_inhib =-70,
                 G_syn_inhib=8,
                 I_bg=4*ones(N_inhib),
                 τ_inhib=70.0)

        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            inhib = HHNeuronInhib(; name=Symbol(:inh, i),
                                  namespace = inner_namespace,
                                  E_syn = E_syn_inhib,
                                  G_syn = G_syn_inhib,
                                  τ = τ_inhib,
                                  I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, namespace, inhibs, graph)
    end
end

"""
    GPe(; N_inhib = 25,
        E_syn_inhib =-70,
        G_syn_inhib=8,
        I_bg=4*ones(N_inhib),
        τ_inhib=70.0
)

A model of Globus Pallidus Externus (GPe). A group of [`HHNeuronInhib`](@ref) neurons, which are typically project to the inhibitory neurons in [`Striatum`](@ref) and to the ones in [`GPi`](@ref).

Arguments : 
- `N_inhib` : Number of inhibitory neurons.
- `E_syn_inhib` [mV, reversal potential for GABA A receptors]
- `G_syn_inhib` [mV, GABA A receptor conductance]
- `I_bg` : [μA] Background current applied to all neurons. If it is a single value then the same current is applied to all `N_inhib` inhibitory neurons. If it is a Vector then it needs to be of length `N_inhib` and each element is applied to one neuron.
- `τ_inhib` : [ms, decay time constant for GABA A receptor conductance]

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.  
"""
struct GPe <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    inhibs::Vector{HHNeuronInhib}
    graph::GraphSystem
    function GPe(;name,
                 namespace=nothing,
                 N_inhib=15,
                 E_syn_inhib=-70,
                 G_syn_inhib=3,
                 I_bg=2*ones(N_inhib),
                 τ_inhib=70.0)
        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        inhibs = map(1:N_inhib) do i
            inhib = HHNeuronInhib(; name=Symbol(:inh, i),
                                  namespace=inner_namespace,
                                  E_syn = E_syn_inhib,
                                  G_syn = G_syn_inhib,
                                  τ = τ_inhib,
                                  I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, namespace, inhibs, graph)
    end
end


"""
    Thalamus(N_exci = 25,
            E_syn_exci=0,
            G_syn_exci=3,
            I_bg=3*ones(N_exci),
            τ_exci=5.0,
            density=0.0,
            weight=1.0,
            rng=default_rng(),
            connection_matrix=subcortical_connection_matrix(density, N_exci, weight; rng)
    )

A Thalamus blox used to model either the Thalamus core or matrix subregions. 
It contains N_exci [`HHNeuronExci`](@ref) neurons connected via a connection_matrix.

Arguments : 
- `N_exci` : Number of excitatory neurons.
- `E_syn_exci` [mV, reversal potential for AMPA receptors]
- `G_syn_exci` [mV, AMPA receptor conductance]
- `I_bg` : [μA] Background current applied to excitatory neurons. If it is a single value then the same current is applied to all `N_exci` excitatory neurons. If it is a Vector then it needs to be of length `N_exci` and each element is applied to one neuron.
- `τ_exci` : [ms, decay time constant for AMPA receptor conductance]
- `density` : Value in range [0,1]. Connection density that determines the sparseness of the Thalamus circuit.
- `weight` : Connection weight to be used in all connections made between the excitatory neurons of Thalamus.
- `rng` : Random number generator. This may be used in the connection_matrix to sample connections based on the density value.
- `connection_matrix` : The connection rule that is applied between the excitatory neurons. By default subcortical_connection_matrix determines whether a connection between every possible pair of [`N_exci` x `N_exci`] neurons is made by flipping a coin with a probability of success equal to the density value. 

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
struct Thalamus <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    excis::Vector{HHNeuronExci}
    graph::GraphSystem
    function Thalamus(;name,
                      namespace=nothing,
                      N_exci = 25,
                      E_syn_exci=0,
                      G_syn_exci=3,
                      I_bg=3*ones(N_exci),
                      τ_exci=5.0,
                      density=0.0,
                      weight=1.0,
                      rng=default_rng(),
                      connection_matrix=subcortical_connection_matrix(density, N_exci, weight; rng))
        graph = GraphSystem()
        inner_namespace = namespaced_name(namespace, name)
        excis = map(1:N_exci) do i
            exci = HHNeuronExci(; name=Symbol(:exci, i),
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
                    add_connection!(graph, excis[i], excis[j], weight=cij)
                end
            end
        end
        new(name, namespace, excis, graph)
    end
end

"""
    STN(; N_exci = 25,
        E_syn_exci=0,
        G_syn_exci=3,
        I_bg=3*ones(N_exci),
        τ_exci=5.0
    )

A model of the Subthalamic Nucleus (STN). A group of [`HHNeuronExci`](@ref) neurons, which are typically project to the inhibitory neurons in [`GPe`](@ref) and to the ones in [`GPi`](@ref).

Arguments : 
- `N_exci` : Number of excitatory neurons.
- `E_syn_exci` [mV, reversal potential for AMPA receptors]
- `G_syn_exci` [mV, AMPA receptor conductance]
- `I_bg` : [μA] Background current applied to excitatory neurons. If it is a single value then the same current is applied to all `N_exci` excitatory neurons. If it is a Vector then it needs to be of length `N_exci` and each element is applied to one neuron.
- `τ_exci` : [ms, decay time constant for AMPA receptor conductance]

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
struct STN <: AbstractComposite
    name::Symbol
    namespace::Union{Nothing, Symbol}
    excis::Vector{HHNeuronExci}
    graph::GraphSystem
    function STN(;name,
                 namespace=nothing,
                 N_exci = 25,
                 E_syn_exci=0,
                 G_syn_exci=3,
                 I_bg=3*ones(N_exci),
                 τ_exci=5.0)
        graph = GraphSystem()
        excis = map(1:N_exci) do i
            exci = HHNeuronExci(; name=Symbol(:exci, i),
                                namespace=namespaced_name(namespace, name),
                                E_syn = E_syn_exci,
                                G_syn = G_syn_exci,
                                τ = τ_exci,
                                I_bg = I_bg[i])
            system_wiring_rule!(graph, exci)
            exci
        end
        new(name, namespace, excis, graph)
    end
end



