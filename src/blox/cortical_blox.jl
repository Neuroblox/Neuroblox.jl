@parameters t
D = Differential(t)

#creates an ODESystem of a cortical block that consist of
# a number of winner-takes-all
function cortical_blox(;name, nblocks=20, blocksize=6)

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
        mat[end,1:end-1].=7;
        mat[1:end-1,end].=1;

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
                        syn[tt[ll],md[jj],] = 1
                    end
                end
            end
        end

        inhib_mod=Nrns;
        syn[inhib,inhib_mod] .= 1;

        return syn, inhib, targ, inhib_mod;
    
    end

    function HH_neuron_wang_excit(;name,E_syn=0.0,G_syn=2,I_in=0,τ=10)
        sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0  

        ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ I_in = I_in


        α_n(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
        β_n(v) = 0.125*exp(-(v+44)/80)

        α_m(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
        β_m(v) = 4*exp(-(v+55)/18)
            
        α_h(v) = 0.07*exp(-(v+44)/20)
        β_h(v) = 1/(1+exp(-(v+14)/10))	

        ϕ = 5 

        G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

        eqs = [ 
                D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+Isyn, 
                D(n)~ϕ*(α_n(V)*(1-n)-β_n(V)*n), 
                D(m)~ϕ*(α_m(V)*(1-m)-β_m(V)*m), 
                D(h)~ϕ*(α_h(V)*(1-h)-β_h(V)*h),
                D(G)~(-1/τ₂)*G + z,
                D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
                ]
        ODESystem(eqs,t,sts,ps;name=name)
    end

    function HH_neuron_wang_inhib(;name,E_syn=0.0,G_syn=2, I_in=0, τ=10)
        sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Iasc(t) = 0.0 Isyn(t)=0.0 G(t)=0 z(t)=0 
        ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = -0 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ 

            α_n(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
            β_n(v) = 0.125*exp(-(v+48)/80)

            α_m(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
            β_m(v) = 4*exp(-(v+58)/18)

            α_h(v) = 0.07*exp(-(v+51)/20)
            β_h(v) = 1/(1+exp(-(v+21)/10))

        ϕ = 5

        G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

        eqs = [ 
                D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+Iasc+Isyn, 
                D(n)~ϕ*(α_n(V)*(1-n)-β_n(V)*n), 
                D(m)~ϕ*(α_m(V)*(1-m)-β_m(V)*m), 
                D(h)~ϕ*(α_h(V)*(1-h)-β_h(V)*h),
                D(G)~(-1/τ₂)*G + z,
                D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
                
                ]

        ODESystem(eqs,t,sts,ps;name=name)
    end

    function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix, input_ar=input_ar,inh_nrn = inh_nrn, inh_mod_nrn = inh_mod_nrn)
        syn_eqs= [ 0~sys[1].V - sys[1].V]

        Nrns = length(adj_matrix[1,:])  

        for ii = 1:length(sys)
            
            presyn = findall(x-> x>0.0, adj_matrix[ii,:])
            wts = adj_matrix[ii,presyn]
            
            presyn_nrn = sys[presyn]
            postsyn_nrn = sys[ii]
                
            if length(presyn)>0
                        
                ind = [i for i = 1:length(presyn)];
                

                    eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*adj[(presyn[p]-1)*Nrns + ii],ind)-postsyn_nrn.Isyn]
                push!(syn_eqs,eq[1])
                
            else
                eq = [0~postsyn_nrn.Isyn];
                push!(syn_eqs,eq[1]);
                
            end

            if inh_mod_nrn[ii]>0
                eq2 = [0 ~ postsyn_nrn.Iasc - input_ar[inh_mod_nrn[ii]]];
                push!(syn_eqs,eq2[1])
            end

            if inh_nrn[ii]>0
                eq2 = [0 ~ postsyn_nrn.Iasc];
                push!(syn_eqs,eq2[1])
            end
            
        end
        popfirst!(syn_eqs)

        @named synaptic_eqs = ODESystem(syn_eqs,t)

        sys_ode = [sys[ii] for ii = 1:length(sys)]

        @named synaptic_network = compose(synaptic_eqs, sys_ode)

        return structural_simplify(synaptic_network)   
    end

    function ascending_input(t,freq,phase,amp=1.4)
        return amp*(sin(t*freq*2*pi/1000-phase+pi/2)+1)
    end

    syn, inhib, targ, inhib_mod = cb_adj_gen(nblocks,blocksize)

    Nrns = length(syn[:,1])
    inh_nrn = zeros(Nrns)
    inh_mod_nrn = zeros(Nrns)
    inh_nrn = convert(Vector{Int64},inh_nrn)
    inh_mod_nrn = convert(Vector{Int64},inh_mod_nrn)
    inh_nrn[inhib] .= 1
    inh_mod_nrn[inhib_mod] = 1

    @parameters adj[1:Nrns*Nrns] = vec(syn)

    amp = 0.3
    freq=16	
    asc_input = ascending_input(t,freq,0,amp);	

    #constant current amplitudes that feed into excitatory (target) neurons

    input_pat = (1 .+ sign.(rand(length(targ)) .- 0.8))/2;
    I_in = zeros(Nrns);
    I_in[targ] .= (4 .+2*randn(length(targ))).*input_pat

    E_syn=zeros(1,Nrns);
    E_syn[inhib] .=-70;
    E_syn[inhib_mod] = -70;

    G_syn=3*ones(1,Nrns);
    G_syn[inhib] .= 11;
    G_syn[inhib_mod] = 40;

    τ = 5*ones(Nrns);
    τ[inhib] .= 70;
    τ[inhib_mod] = 70;

    nrn_network=[]
    for ii = 1:Nrns
        if (inh_nrn[ii]>0) || (inh_mod_nrn[ii]>0)
    nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
            
        else

    nn = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
        end
    push!(nrn_network,nn)
    end

    @named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn, input_ar=asc_input, inh_nrn = inh_nrn, inh_mod_nrn=inh_mod_nrn)

end
