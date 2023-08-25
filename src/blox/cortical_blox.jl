@parameters t
D = Differential(t)

#creates an ODESystem of a cortical block that consist of
# a number of winner-takes-all
function cortical_blox(;name, nblocks=6, blocksize=6)

    #creates weight matrix for cortical block of given number of wta blocks : nblocks
    #                                                and size of each block : blocksize
    #returns : 
    # 	      syn : weight matrix of size Nrns
    # 		  inhib: indices of feedback inhibitory neurons
    # 		  targ: indices of excitatory (target) neurons
    # 		  inhib_mod: indices of modulatory inhibitory neurons

    function cb_adj_gen(nblocks = 16, blocksize = 6)
        Nrns = blocksize*nblocks+1;

        #block
        mat = zeros(blocksize,blocksize);
        mat[1:end-1,end].=7;
        mat[end,1:end-1].=1;

        #disjointed blocks
        syn = zeros(Nrns,Nrns);
        for ii = 1:nblocks;
            syn[(ii-1)*blocksize+1:(ii*blocksize),(ii-1)*blocksize+1:(ii*blocksize)] = mat;
        end

        inhib = [kk*blocksize for kk = 1:nblocks]

        tot = [kk for kk=1:(Nrns-1)]
        targ = setdiff(tot,inhib);

        for ii = 1:nblocks
            md = [kk for kk = 1+(ii-1)*blocksize : ii*blocksize];
            tt = setdiff(targ,md);
            
            for jj = 1:blocksize-1
                
                for ll = 1:length(tt)
                    rr = rand()
                    if rr <= 1/length(tt)
                        syn[md[jj],tt[ll]] = 1
                    end
                end
            end
        end

        inhib_asc=Nrns;
        syn[inhib_asc,inhib] .= 1;

        return syn, inhib, targ, inhib_asc;
    
    end
   
    syn, inhib, targ, inhib_asc = cb_adj_gen(nblocks,blocksize)

    Nrns = length(syn[:,1])
    inh_nrn = zeros(Nrns)
    inh_asc_nrn = zeros(Nrns)
    inh_nrn[inhib] .= 1
    inh_asc_nrn[inhib_asc] = 1

    #constant current amplitudes that feed into excitatory (target) neurons

    input_pat = (1 .+ sign.(rand(length(targ)) .- 0.8))/2
    I_in = zeros(Nrns)
    I_in[targ] .= (4 .+2*randn(length(targ))).*input_pat
    I_in[inhib_asc] = 0.3

    freq = zeros(Nrns)
    freq[inhib_asc] = 16

    G_syn=3*ones(Nrns);
    G_syn[inhib] .= 11;
    G_syn[inhib_asc] = 40;

   
    nrn_network=[]
    for ii = 1:Nrns
        if (inh_nrn[ii]>0) || (inh_asc_nrn[ii]>0)
           nn = HHNeuronInhibBlox(name=Symbol("nrn$ii"),G_syn=G_syn[ii],I_in=I_in[ii],freq=freq[ii]) 
        else
           nn = HHNeuronExciBlox(name=Symbol("nrn$ii"),G_syn=G_syn[ii],I_in=I_in[ii],freq=freq[ii]) 
        end
        push!(nrn_network,nn)
    end
    
    sys = [s.odesystem for s in nrn_network]
    connect = [s.connector for s in nrn_network] 
    
    @named syn_net = SynapticConnections(sys=sys, adj_matrix=syn, connector=connect)
    return syn_net, syn
end

struct CorticalBlox{N, P, S, C} <: AbstractComponent
    namespace::N
    parts::Vector{P}
    odesystem::S
    connector::C
    P_connect::Float64

    function CorticalBlox(;
        name, 
        namespace = nothing,
        N_exci = 5,
        N_wta = 2,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=3.0,
        freq=zeros(N_exci),
        phase=zeros(N_exci),
        τ_exci=5,
        τ_inhib=70
    )
        
        P_connect = 1 / (N_exci * N_wta)

        wtas = map(Base.OneTo(N_wta)) do i
            WinnerTakeAllBlox(;
                name = Symbol("wta$i"), 
                namespace = namespaced_name(namespace, name),
                N_exci = 5,
                P_connect,
                E_syn_exci,
                E_syn_inhib,
                G_syn_exci,
                G_syn_inhib,
                I_in = rand(N_exci),
                freq,
                phase,
                τ_exci,
                τ_inhib    
            )
        end

        g = MetaDiGraph()
        add_blox!.(Ref(g), wtas)

        idxs = Base.OneTo(N_wta)
        for i in idxs
            add_edge!.(Ref(g), i, setdiff(idxs, i), :weight, 1.0)
        end

        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(wtas; name)

        new{typeof(namespace), eltype(wtas), typeof(sys), typeof(bc)}(namespace, wtas, sys, bc)
    end
end

