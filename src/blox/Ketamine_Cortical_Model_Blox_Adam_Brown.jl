"""
cortical neuronal assembly used in ketamine model in Adam et al,2024 
"""

struct Cortical_Pyramidal_Assembly_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function Cortical_Pyramidal_Assembly_Adam(;
        name, 
        namespace = nothing,
        N_exci = 80,
        E_syn_exci=0,
        I_bg=-0.25*ones(N_exci),
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=1.5,
        σ=3.0,
        density=10/80,
        weight=0.2
    )
        n_exci = [
            HHNeuronExci_pyr_Adam_Blox(
                    name = Symbol("exci$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_exci, 
                    τ = τ_exci,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ
            ) 
            for i in Base.OneTo(N_exci)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_exci)
            add_blox!(g, n_exci[i])
        end
        in_degree = Int(ceil(density*(N_exci)))
        idxs = Base.OneTo(N_exci)
        for i in idxs
            source_set = setdiff(idxs, i)
            source = sample(source_set, in_degree; replace=false)
            for j in source
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
        end

        parts = n_exci

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    



struct Cortical_Interneuron_Assembly_Phasic_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function Cortical_Interneuron_Assembly_Phasic_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 20,
        E_syn_inhib=-80.0,
        I_bg= 0.1*ones(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=6,
        σ=3.0,
        density=10/20,
        weight=0.8
    )
        n_inh = [
            HHNeuronInh_inter_Adam_Blox(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_inhib)
            add_blox!(g, n_inh[i])
        end
        in_degree = Int(ceil(density*(N_inhib)))
        idxs = Base.OneTo(N_inhib)
        for i in idxs
            source_set = setdiff(idxs, i)
            source = sample(source_set, in_degree; replace=false)
            for j in source
                add_edge!(g, j, i, Dict(:weight=>weight/in_degree))
            end
        end

        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    


struct Cortical_Interneuron_Assembly_Tonic_Adam <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    mean

    function Cortical_Interneuron_Assembly_Tonic_Adam(;
        name, 
        namespace = nothing,
        N_inhib = 80,
        E_syn_inhib=-80.0,
        I_bg= -1.4*ones(N_inhib),
        freq=zeros(N_inhib),
        phase=zeros(N_inhib),
        τ_inhib=8,
        σ=3.0,
        density=0/20,
        weight=5
    )
        n_inh = [
            HHNeuronInh_inter_Adam_Blox(
                    name = Symbol("inh$i"),
                    namespace = namespaced_name(namespace, name), 
                    E_syn = E_syn_inhib, 
                    τ = τ_inhib,
                    I_bg = I_bg[i],
                    freq = freq[i],
                    phase = phase[i],
                    σ=σ,
                    N_nmda=1
            ) 
            for i in Base.OneTo(N_inhib)
        ]

        g = MetaDiGraph()
        for i in Base.OneTo(N_inhib)
            add_blox!(g, n_inh[i])
        end
        
        parts = n_inh

        bc = connector_from_graph(g)
    
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(parts; name)
        
        m = if isnothing(namespace) 
            [s for s in unknowns.((sys,), unknowns(sys)) if contains(string(s), "V(t)")]
        else
            @variables t
            sys_namespace = System(Equation[], t; name=namespaced_name(namespace, name))
            [s for s in unknowns.((sys_namespace,), unknowns(sys)) if contains(string(s), "V(t)")]
        end

        new(namespace, parts, sys, bc, m)
    end

end    
struct NMDA_receptor <: ReceptorBlox 
    namespace
    odesystem

    function NMDA_receptor(;
        name,
        namespace = nothing,
    )

        sts = @variables begin 
            Glu(t)=0.0
            [input=true] 
            V(t)=-67.0
            [input=true] 
            C(t)=0.5
            C_A(t)=0.0
            C_AA(t)=0.0
            D_AA(t)=0.0
            O_AA(t)=0.0
            [output = true] 
            O_AAB(t)=0.0
            C_AAB(t)=0.0
            D_AAB(t)=0.0
            C_AB(t)=0.0
            C_B(t)=0.5     
        end

        ps = @parameters begin 
            k_on=5
            k_off=0.0055
            k_r=0.0018
            k_d=0.0084
            k_unblock=5.4
            k_block=0.61
            α=0.0916
            β=0.0465
        end

        eqs = [
                D(C) ~  k_off*C_A - 2*k_on*Glu*C,
                D(C_A) ~ 2*k_off*C_AA +  2*k_on*Glu*C - (k_on*Glu + k_off)*C_A,
                D(C_AA) ~ k_on*Glu*C_A + α*O_AA + k_r*D_AA - (2*k_off + β + k_d)*C_AA,
                D(D_AA) ~ k_d*C_AA - k_r*D_AA,
                D(O_AA) ~ β*C_AA + k_unblock*exp(V/47)*O_AAB - (α + k_block*exp(-V/17))*O_AA,
                D(O_AAB) ~ k_block*exp(-V/17)*O_AA + β*C_AAB - (k_unblock*exp(V/47) + α)*O_AAB,
                D(C_AAB) ~ α*O_AAB + k_on*Glu*C_AB + k_r*D_AAB - (β + 2*k_off + k_d)*C_AAB,
                D(D_AAB) ~ k_d*C_AAB - k_r*D_AAB,
                D(C_AB) ~ 2*k_off*C_AAB + 2*k_on*Glu*C_B - (k_on*Glu + k_off)*C_AB,
                D(C_B) ~ k_off*C_AB - 2*k_on*Glu*C_B

        ]
        sys = System(
                eqs, t, sts, ps; 
			    name = Symbol(name)
			    )
        new(namespace, sys)
    end
end

struct Steady_Glutamate <: StimulusBlox
    namespace
    odesystem

    function Steady_Glutamate(;
        name, 
        namespace = nothing,
        glu = 1.0,
        E_syn = 0.0      
    )
        sts = @variables Glu(t) = glu
        ps = @parameters E_syn = E_syn, glu=glu

        eqs = [Glu ~ glu]
        sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)
        new(namespace, sys)
    end
end

struct Glutamate_puff <: StimulusBlox
    namespace
    odesystem

    function Glutamate_puff(;
        name, 
        namespace = nothing,
        glu = 1.0,
        E_syn = 0.0,
        τ_glu = 1.2      
    )
        sts = @variables Glu(t) = glu
        ps = @parameters E_syn = E_syn, glu=glu, τ_glu=τ_glu

        eqs = [Glu ~ -Glu/τ_glu]
        sys = System(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)
        new(namespace, sys)
    end
end