"""
    Cortical(; N_wta=20, N_exci=5, E_syn_exci=0, E_syn_inhib=-70, G_syn_exci=3, G_syn_inhib=3,, G_syn_ff_inhib=3.5, I_bg_ar=0, τ_exci=5, τ_inhib=70)

A component representing a layer of neocortex. It contains multiple lateral inhibition `WinnerTakeAll` microcircuits and a single feedforward inhibitory neuron.
Excitatory neurons across `WinnerTakeAll` microcircuits are connected together using a hypergeometric rule [1]. The feedforward neuron inhibits each one of the `N_wta` local interneurons. 
The excitatory neurons are [`HHNeuronExci`](@ref) and the feedforward and feedback interneurons are [`HHNeuronInhib`](@ref) and their parameters are based on [2].

Arguments : 
- `N_wta` : Number of `WinnerTakeAll` microcircuits in the cortical circuit.
- `N_exci` : Number of excitatory neurons in each one of the `N_wta` `WinnerTakeAll` components.
- `E_syn_exci` [mV, reversal potential for AMPA receptors]
- `E_syn_inhib` [mV, reversal potential for GABA A receptors]
- `G_syn_exci` [mV, AMPA receptor conductance]
- `G_syn_inhib` [mV, GABA A receptor conductance for lateral inhibition]
- `G_syn_ff_inhib` [mV, GABA A receptor conductance for feedforward inhibition]
- `I_bg_ar` : [μA] Background current applied to excitatory neurons. If it is a single value then the same current is applied to all [`N_wta` x `N_exci`] excitatory neurons. If it is a Vector then it needs to be of length `N_exci` and each element is applied to each neuron of every WinnerTakeAll microcircuit.
- `τ_exci` : [ms, decay time constant for AMPA receptor conductance]
- `τ_inhib` : [ms, decay time constant for GABA A receptor conductance]

References: 
1. Andrew C. Felch and Richard H. Granger. The hypergeometric connectivity hypothesis: Divergent performance of brain circuits with different synaptic connectivity distributions. Brain Research, 1202:3–13, April 2008.
2. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
struct Cortical <: AbstractComposite
    name::Symbol
    namespace::Union{Symbol, Nothing}
    wtas
    n_ff_inh
    graph::GraphSystem
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
                      τ_exci=5.0,
                      τ_inhib=70.0,
                      kwargs...)

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
        g = GraphSystem(;name=namespaced_name(namespace, name))
        @graph! g begin
            @nodes begin
                ff_inh = HHNeuronInhib(E_syn=E_syn_inhib, G_syn=G_syn_ff_inhib, τ=τ_inhib)
                wtas = for i ∈ 1:N_wta
                    if I_bg_ar isa AbstractArray
                        I_bg = I_bg_ar[i]
                    else
                        I_bg = I_bg_ar
                    end
                    WinnerTakeAll(; name=Symbol("wta$i"),
                                  # namespace=g.name,
                                  N_exci,
                                  E_syn_exci,
                                  E_syn_inhib,
                                  G_syn_exci,
                                  G_syn_inhib,
                                  I_bg = I_bg,
                                  τ_exci,
                                  τ_inhib)
                end
            end
            @connections begin
                for i ∈ 1:N_wta
                    # connect the inhibitory neuron to the i-th wta
                    ff_inh => wtas[i], [weight=1.0]
                    for j ∈ 1:N_wta
                        if j != i
                            wtas[i] => wtas[j], [kwargs..., connection_matrix=connection_matrices[i, j]]
                        end
                    end
                end
            end
        end
        new(name, namespace, wtas, ff_inh, g)
    end
end

"""
    WinnerTakeAll(; N_exci=5, E_syn_exci=0, E_syn_inhib=-70, G_syn_exci=3, G_syn_inhib=3, I_bg=zeros(N_exci), τ_exci=5, τ_inhib=70)

A winner-take-all microcircuit found in neocortex, also known as a lateral inhibition microcircuit [1].
Multiple pyramidal (excitatory) neurons send synapses to a single interneuron (inhibitory) and receive feedback inhibition from that interneuron.
The excitatory neurons are [`HHNeuronExci`](@ref) and and feedback interneuron is [`HHNeuronInhib`](@ref) and their parameters are based on [2].

Arguments : 
- `N_exci` : Number of excitatory neurons.
- `E_syn_exci` [mV, reversal potential for AMPA receptors]
- `E_syn_inhib` [mV, reversal potential for GABA A receptors]
- `G_syn_exci` [mV, AMPA receptor conductance]
- `G_syn_inhib` [mV, GABA A receptor conductance]
- `I_bg` : [μA] Background current applied to excitatory neurons. If it is a single value then the same current is applied to all `N_exci` excitatory neurons. If it is a Vector then it needs to be of length `N_exci` and each element is applied to each neuron.
- `τ_exci` : [ms, decay time constant for AMPA receptor conductance]
- `τ_inhib` : [ms, decay time constant for GABA A receptor conductance]

References: 
1. Coultrip, Robert, Richard Granger, and Gary Lynch. “A Cortical Model of Winner-Take-All Competition via Lateral Inhibition.” Neural Networks 5, no. 1 (January 1, 1992): 47-54.
2. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
struct WinnerTakeAll <: AbstractComposite
    name::Symbol
    namespace::Union{Symbol, Nothing}
    inhi::HHNeuronInhib
    excis::Vector{HHNeuronExci}
    graph::GraphSystem
    function WinnerTakeAll(;name,
                           namespace=nothing,
                           N_exci = 5,
                           E_syn_exci=0.0,
                           E_syn_inhib=-70,
                           G_syn_exci=3.0,
                           G_syn_inhib=3.0,
                           I_bg=zeros(N_exci),
                           # phase=0.0,
                           τ_exci=5.0,
                           τ_inhib=70.0)

        g = GraphSystem(; name= namespaced_name(namespace, name))
        @graph! g begin
            @nodes begin
                inhi = HHNeuronInhib(name = :inh,
                                     namespace = namespaced_name(namespace, name),
                                     E_syn = E_syn_inhib, G_syn = G_syn_inhib, τ = τ_inhib)
                
                excis = for i ∈ 1:N_exci
                    HHNeuronExci(
                        name = Symbol("exci$i"),
                        I_bg = (I_bg isa AbstractArray) ? I_bg[i] : I_bg*rand(), # behave differently if I_bg is array
                        E_syn = E_syn_exci,
                        G_syn = G_syn_exci,
                        τ = τ_exci
                    )
                end
            end
            @connections begin
                for excii ∈ excis
                    inhi => excii, [weight=1.0]
                    excii => inhi, [weight=1.0]
                end
            end
        end
        new(name, namespace, inhi, excis, g)
    end
end

get_ff_inh_neurons(n::AbstractInhNeuron) = [n]
get_ff_inh_neurons(n) = AbstractInhNeuron[]
get_ff_inh_neurons(b::Cortical) = [b.n_ff_inh]
