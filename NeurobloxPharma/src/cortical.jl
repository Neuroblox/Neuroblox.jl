struct Cortical <: AbstractComposite
    namespace
    parts
    system
    connector
    connection_matrices
    kwargs

    function Cortical(;
        name, 
        N_wta=20,
        namespace=nothing,
        N_exci=5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=4.0,
        G_syn_ff_inhib=3.5,
        I_bg_ar=0,
        I_bg_ff_inhib=0,
        τ_exci=5,
        τ_inhib=70,             
        kwargs...
            )
        
        wtas = map(1:N_wta) do i
            if I_bg_ar isa Array
                I_bg = I_bg_ar[i]
            else
                I_bg = I_bg_ar
            end
            WinnerTakeAll(;
                name=Symbol("wta$i"),
                namespace=namespaced_name(namespace, name),
                N_exci,
                E_syn_exci,
                E_syn_inhib,
                G_syn_exci,
                G_syn_inhib,
                I_bg = I_bg,
                τ_exci,
                τ_inhib,
                rng=get(kwargs, :rng, default_rng())          
            )
        end
        n_ff_inh = HHNeuronInhib(
            name = "ff_inh",
            namespace = namespaced_name(namespace, name),
            E_syn = E_syn_inhib,
            G_syn = G_syn_ff_inhib,
            I_bg = I_bg_ff_inhib,
            τ = τ_inhib
        )
        # users can supply a matrix of connection matrices.
        # connection_matrices[i,j][k, l] determines if neuron k from wta i is connected to
        # neuron l from wta j.
        connection_matrices = get(kwargs, :connection_matrices) do
            map(Iterators.product(1:N_wta, 1:N_wta)) do (i, j)
                get_connection_matrix(kwargs,
                                      Symbol("wta$i"), Symbol("wta$j"),
                                      N_exci, N_exci)
            end
        end
        g = MetaDiGraph()
        add_blox!.(Ref(g), vcat(wtas, n_ff_inh))
        for i in 1:N_wta
            for j in 1:N_wta
                if j != i
                    kwargs_ij = merge(kwargs, Dict(:connection_matrix => connection_matrices[i, j]))
                    add_edge!(g, i, j, kwargs_ij)
                end
            end
            add_edge!(g, N_wta+1, i, Dict(:weight => 1))
        end
        
        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the Connector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(vcat(wtas, n_ff_inh); name)

        new(namespace, vcat(wtas, n_ff_inh), sys, bc, connection_matrices, kwargs)
    end
end

"""
    WinnerTakeAll

Creates a winner-take-all local circuit found in neocortex,
typically 5 pyramidal (excitatory) neurons send synapses to a single interneuron (inhibitory)
and receive feedback inhibition from that interneuron.

References: 
- Coultrip, Robert, Richard Granger, and Gary Lynch. “A Cortical Model of Winner-Take-All Competition via Lateral Inhibition.” Neural Networks 5, no. 1 (January 1, 1992): 47-54.
- Pathak A., Brincat S., Organtzidis H., Strey H., Senneff S., Antzoulatos E., Mujica-Parodi L., Miller E., Granger R. "Biomimetic model of corticostriatal micro-assemblies discovers new neural code.", bioRxiv 2023.11.06.565902, 2024
"""
struct WinnerTakeAll{P} <: AbstractComposite
    namespace
    parts::Vector{P}
    system
    connector

    function WinnerTakeAll(; 
        name, 
        namespace = nothing,
        N_exci = 5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=3.0,
        I_bg=zeros(N_exci),
        τ_exci=5,
        τ_inhib=70,
        rng=default_rng()                   
    )  
        n_inh = HHNeuronInhib(
            name = "inh",
            namespace = namespaced_name(namespace, name), 
            E_syn = E_syn_inhib, 
            G_syn = G_syn_inhib, 
            τ = τ_inhib
        )
        n_excis = [
            HHNeuronExci(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    G_syn = G_syn_exci, 
                    τ = τ_exci,
                    I_bg = (I_bg isa Array) ? I_bg[i] : I_bg*rand(rng), # behave differently if I_bg is array
            ) 
            for i in Base.OneTo(N_exci)
        ]

        g = MetaDiGraph()
        add_blox!(g, n_inh)
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_excis[i])
            add_edge!(g, 1, i+1, :weight, 1.0)
            add_edge!(g, i+1, 1, :weight, 1.0)
        end

        parts = vcat(n_inh, n_excis)
        
        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)

        new{Union{eltype(n_excis), typeof(n_inh)}}(namespace, parts, sys, bc)
    end 
end


