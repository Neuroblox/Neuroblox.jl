### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 0626a6e2-bc63-11ec-38cf-4dd06e14134c
import Pkg

# ╔═╡ 275a05c4-ae88-4c00-a05b-d1fd80eabdee
Pkg.activate(".")

# ╔═╡ 8e35f55b-6973-4f9e-95f5-01012501a3d2
using DSP

# ╔═╡ d99586a3-0a59-4613-a8af-367532245262
using ModelingToolkit, OrdinaryDiffEq

# ╔═╡ b1c628ac-d6bd-46e7-84c2-55edf2395c1c
using DifferentialEquations

# ╔═╡ fc99f083-41d1-4e67-acf9-183b23dab8ee
using Plots

# ╔═╡ 6d28d8c2-1527-42fd-907d-1c42c3ada4da
using Symbolics

# ╔═╡ d12cea97-8b58-4ed1-a94c-47ae313c6a03
using Distributions, Random

# ╔═╡ 69675627-5794-4d3d-9f3b-f0f66a281989
using Colors

# ╔═╡ 0c99623d-3ed1-4974-a71b-3421b9af6024
using Images

# ╔═╡ 430a595a-649c-45c8-939d-13cb7b5eb307
@variables t

# ╔═╡ 792a219b-373b-4c3f-a960-312dbd06d546
D = Differential(t)

# ╔═╡ 670ea12e-6940-412e-93cb-e463ed12ddea
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

# ╔═╡ 3aac2fa5-8895-4a44-9402-2b3644f98d1e
#defining weight matrix for single cortical block

begin

	nblocks=20
	blocksize=6
	
	syn, inhib, targ, inhib_mod = cb_adj_gen(nblocks,blocksize)
	
	Nrns = length(syn[:,1])
	inh_nrn = zeros(Nrns)
	inh_mod_nrn = zeros(Nrns)
	inh_nrn = convert(Vector{Int64},inh_nrn)
	inh_mod_nrn = convert(Vector{Int64},inh_mod_nrn)
	inh_nrn[inhib] .= 1
	inh_mod_nrn[inhib_mod] = 1

	@parameters adj[1:Nrns*Nrns] = vec(syn)


	
end

# ╔═╡ 70391271-612a-4840-919e-0c7db60afeef
#Required functions (these will go in blocks folder)


# defining two types of neurons

begin

function HH_neuron_wang_excit(;name,E_syn=0.0,G_syn=2,I_in=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 I_syn(t)=0.0 G(t)=0.0 z(t)=0.0  
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ I_in = I_in
	
	
 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+I_syn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end

function HH_neuron_wang_inhib(;name,E_syn=0.0,G_syn=2, I_in=0, τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Iasc(t) = 0.0 I_syn(t)=0.0 G(t)=0 z(t)=0 
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = -0 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ 
	
		αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
		βₙ(v) = 0.125*exp(-(v+48)/80)

		αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
		βₘ(v) = 4*exp(-(v+58)/18)

		αₕ(v) = 0.07*exp(-(v+51)/20)
		βₕ(v) = 1/(1+exp(-(v+21)/10))

	
ϕ = 5
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+Iasc+I_syn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
	       
	      ]
	
	ODESystem(eqs,t,sts,ps;name=name)
end


# function that constructs ode system for entire network given array of single neuron odesystems: sys, adjacency/weight  matrix: adj_matrix, array of modulatory inputs: input_ar (one for each block), indices of inhibitory neurons: inh_nrn, indices of modulatory inhibitory neurons: inh_mod_nrn 	

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
	       

			 eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*adj[(presyn[p]-1)*Nrns + ii],ind)-postsyn_nrn.I_syn]
            push!(syn_eqs,eq[1])
			
		else
		    eq = [0~postsyn_nrn.I_syn];
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
	
end	

# ╔═╡ d014b429-efdf-4e6b-b0ae-a14bd6ec67df
#defining inputs

#modulatory input that feeds into modulatory inhibitory neuron
begin

function ascending_input(t,freq,phase,amp=1.4)

	return amp*(sin(t*freq*2*pi/1000-phase+pi/2)+1)
end
amp = 0.3
freq=16	
asc_input = ascending_input(t,freq,0,amp);	


#constant current amplitudes that feed into excitatory (target) neurons

input_pat = (1 .+ sign.(rand(length(targ)) .- 0.8))/2;
I_in = zeros(Nrns);
I_in[targ] .= (4 .+2*randn(length(targ))).*input_pat

end	

# ╔═╡ 83a02d9c-645c-4dea-9c90-3c1cba6f8c9c
#Parameter set for each neuron

begin

    simtime = 2000 #simulation stime in ms
	
    E_syn=zeros(1,Nrns);
	E_syn[inhib] .=-70;
	E_syn[inhib_mod] = -70;



	G_syn=3*ones(1,Nrns);
	G_syn[inhib] .= 11;
	G_syn[inhib_mod] = 40;

		
	τ = 5*ones(Nrns);
	τ[inhib] .= 70;
	τ[inhib_mod] = 70;

end

# ╔═╡ 84410723-dcb4-46eb-bac4-871aac88cf8c
#construct odesystems for each neuron and then construct the composite system for entire network using function synaptic_network() 
begin

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

# ╔═╡ a496e9c1-8a3c-4734-b414-c6469cb1e193
plot(Gray.(syn/5))

# ╔═╡ 9548e8bb-bdbc-4c18-9a6d-98a808e763f9
prob = ODEProblem(syn_net, [], (0, simtime));

# ╔═╡ d9c4003f-4392-4aab-8964-035ad2195b7b
sol = solve(prob,Vern7(),saveat = 0.1)#,saveat = 0.1,reltol=1e-4,abstol=1e-4);

# ╔═╡ 263b26c2-b19b-4889-8e8c-fa3aef952649
# this extracts indices for input current amplitudes I_in and for weight matrix elements adj from the parameters of the entire system. These are usefull for
#changing the input currents and weights of already existing connections by remaking #the odeprob 
begin
indexof(sym,syms) = findfirst(isequal(sym),syms)
global	cc=[]
for ii in targ
	 vvv = nrn_network[ii].I_in
global	cc=push!(cc,indexof(vvv,parameters(syn_net)))
end

global	dd=[]
	syn_v=vec(syn)
	length(syn_v)
	in_con = findall(x-> x>0,syn_v)
for ii in in_con
	vvv2 = adj[ii]
global	dd=push!(dd,indexof(vvv2,parameters(syn_net)))
end
	
end

# ╔═╡ 66c9c977-7405-4e0d-9807-364404efaa7c
#extracting membrane voltage values from solution
begin
    ss = convert(Array,sol);
	VV=zeros(Nrns,length(sol.t));  V=zeros(Nrns,length(sol.t)); G=zeros(Nrns,length(sol.t));
	
	for ii = 1:Nrns
		
	   	V[ii,:] =  ss[(((ii-1)*6)+1),1:end]; #actual voltage values
		VV[ii,:] = ss[(((ii-1)*6)+1),1:end].+(ii-1)*200; #voltage values for plotting raster plos 
	end

	
  
   average = mean(V[targ,:],dims=1); #summary signal
  


	
end;

# ╔═╡ 5c4d0d30-552f-4b37-a162-c376081f23be
begin
plot(sol.t,VV[targ[1:end],:]',legend=false, color = "blue"); #excitatory neurons
plot!(sol.t,VV[inhib_mod,:], color = "green"); #modulatory inhibitory neuron
plot!(sol.t,VV[inhib[1:20],:]', color = "red",yticks=[],size=(1000,1000)) #inhibitory neurons
end

# ╔═╡ 6126e037-058a-4fd9-bd10-80b377566de7
begin
plot(sol.t,average',ylims=(-69,-60));
end

# ╔═╡ f9286145-4636-4e18-9bb9-756f1e287349
begin
    fs = 1000;
	avec = [ii*10+1 for ii = 1000:2000]
	
	periodogram_estimation = periodogram(average[1,avec], fs=fs)
        pxx = periodogram_estimation.power
        f = periodogram_estimation.freq

end

# ╔═╡ 465ae552-93dc-4f90-ba75-0fc7d58ac609
begin
plot(f[2:50],pxx[2:50]);
end

# ╔═╡ Cell order:
# ╠═0626a6e2-bc63-11ec-38cf-4dd06e14134c
# ╠═275a05c4-ae88-4c00-a05b-d1fd80eabdee
# ╠═8e35f55b-6973-4f9e-95f5-01012501a3d2
# ╠═d99586a3-0a59-4613-a8af-367532245262
# ╠═b1c628ac-d6bd-46e7-84c2-55edf2395c1c
# ╠═fc99f083-41d1-4e67-acf9-183b23dab8ee
# ╠═6d28d8c2-1527-42fd-907d-1c42c3ada4da
# ╠═d12cea97-8b58-4ed1-a94c-47ae313c6a03
# ╠═69675627-5794-4d3d-9f3b-f0f66a281989
# ╠═0c99623d-3ed1-4974-a71b-3421b9af6024
# ╠═430a595a-649c-45c8-939d-13cb7b5eb307
# ╠═792a219b-373b-4c3f-a960-312dbd06d546
# ╠═70391271-612a-4840-919e-0c7db60afeef
# ╠═670ea12e-6940-412e-93cb-e463ed12ddea
# ╠═3aac2fa5-8895-4a44-9402-2b3644f98d1e
# ╠═d014b429-efdf-4e6b-b0ae-a14bd6ec67df
# ╠═83a02d9c-645c-4dea-9c90-3c1cba6f8c9c
# ╠═84410723-dcb4-46eb-bac4-871aac88cf8c
# ╠═a496e9c1-8a3c-4734-b414-c6469cb1e193
# ╠═9548e8bb-bdbc-4c18-9a6d-98a808e763f9
# ╠═d9c4003f-4392-4aab-8964-035ad2195b7b
# ╠═263b26c2-b19b-4889-8e8c-fa3aef952649
# ╠═66c9c977-7405-4e0d-9807-364404efaa7c
# ╠═5c4d0d30-552f-4b37-a162-c376081f23be
# ╠═6126e037-058a-4fd9-bd10-80b377566de7
# ╠═465ae552-93dc-4f90-ba75-0fc7d58ac609
# ╠═f9286145-4636-4e18-9bb9-756f1e287349