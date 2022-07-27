### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 406f6214-cb40-11ec-037a-1325bda2f580
import Pkg

# ╔═╡ 1a01f8a2-d779-4b64-9401-3e746acdd6ab
Pkg.activate(".")

# ╔═╡ dbd16f92-8b0a-49c7-8bfd-1503967bdd9d
using ModelingToolkit

# ╔═╡ c1ee3eed-5730-4ab1-a012-af6cce952024
using DifferentialEquations

# ╔═╡ 407ed4ff-9fec-4df8-a0ba-c264d1f7d2db
using OrdinaryDiffEq

# ╔═╡ 5099a19c-0a25-4dc1-8442-ddf9ac56ef8f
using StochasticDiffEq

# ╔═╡ 72112d41-4432-4233-9ab3-d9011674a3f8
using Plots

# ╔═╡ f7bb61b5-70f1-46ed-a8fd-bb26ca8fc32f
using Distributions

# ╔═╡ 8e6fcff1-3387-42b5-8d1f-8ba769adf6ca
using Statistics

# ╔═╡ 544f27bc-077a-488f-b9a4-8f4ca4cace4b
using Colors

# ╔═╡ 7b070751-5d29-4f97-b4e0-899e35aa7041
using DelimitedFiles

# ╔═╡ 697586f1-0539-474f-99df-4106e39012ba
using Random

# ╔═╡ 4abaf4c3-14ac-4c82-a812-3fd4ee87e824
using Printf

# ╔═╡ 0a803feb-3dd1-43ac-9afc-1b0afd19ce2d
include("Findpeaks.jl")

# ╔═╡ 738fb9f1-81f3-4738-a8bd-407461c9586f
@variables t

# ╔═╡ ca25e5b5-9c81-461f-b014-54221ffd06c6
D = Differential(t)

# ╔═╡ e4b89f6a-21f0-42ca-950c-448afa5ec535
function ascending_input(t,freq,phase,amp=0.65)

	return amp*(sin(t*freq*2*pi/1000-phase+pi/2)+1)
end

# ╔═╡ 61c5b42a-8723-4334-a3ba-8c8558b11284
function HH_neuron_wang_excit(;name,E_syn=0.0,G_syn=2,I_in=0,freq=0,phase=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0  
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ I_in = I_in freq=freq phase=phase
	
	
 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in*(sin(t*freq*2*pi/1000+phase)+1)+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end
	

# ╔═╡ 3be21966-09e5-46be-995c-c53e49d0a3c2
function HH_neuron_wang_inhib(;name,E_syn=0.0,G_syn=2, I_in=0, τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Iasc(t) = 0.0 Isyn(t)=0.0 G(t)=0 z(t)=0 
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
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+Iasc+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
	       
	      ]
	
	ODESystem(eqs,t,sts,ps;name=name)
end
	

# ╔═╡ ae38608c-2193-4439-b439-29fa7805c05f
function cb_adj_gen(nblocks = 16, blocksize = 6)

	
	Nrns = blocksize*nblocks;

	#block
	mat = zeros(blocksize,blocksize);
	mat[end,1:end-1].=1;
	mat[1:end-1,end].=1;

	#disjointed blocks
	syn = zeros(Nrns,Nrns);
    for ii = 1:nblocks;
       syn[(ii-1)*blocksize+1:(ii*blocksize),(ii-1)*blocksize+1:(ii*blocksize)] = mat;
    end


	#connecting the blocks to form pool
	expt =[1]
	for ii = 1:nblocks
		push!(expt,6*ii)
	end
	#push![expt, inhib]
	inhib = [kk*blocksize for kk = 1:nblocks]
	#inhib = expt[2:end]
    tot = [kk for kk=1:Nrns]
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

	#inh_nrn=zeros(Nrns)
	#inh_nrn[inhib] .= 1;

	return syn, inhib, targ;
  
end


# ╔═╡ 77b3c1e6-1ace-4a89-a364-ecd732a7b995
function connect_cb(block_ar, inhib_ar, targ_ar)
	n_block = length(block_ar)
	l = zeros(n_block)
	inhib = inhib_ar[1]
	for ii = 1:n_block
		mat = block_ar[1];
		l[ii] = length(mat[:,1])
		if ii>1
		    targ_ar[ii] = targ_ar[ii] .+ sum(l[1:(ii-1)])
			inhib_ar[ii] = inhib_ar[ii] .+ sum(l[1:(ii-1)])
			inhib = vcat(inhib,inhib_ar[ii])
			
		end
	end
	l = convert(Vector{Int64},l)
	Nrns = sum(l)
	syn = zeros(Nrns,Nrns)
	inh_nrn = zeros(Nrns)
	inh_nrn = convert(Vector{Int64},inh_nrn)
	for jj = 1:n_block
	    if jj==1
		 chk = 0
		else
		 chk = sum(l[1:(jj-1)])
		end
		
		syn[(chk+1):(chk+l[jj]),(chk+1):(chk+l[jj])] = block_ar[jj]

		if jj<n_block

			lt1 = length(targ_ar[jj])
			lt2 = length(targ_ar[jj+1])
			for kk = 1:lt2
				ind2 = randperm(lt1)
				#syn[targ_ar[jj+1][ind2[1:4]],targ_ar[jj][kk]] .= 3
				syn[targ_ar[jj+1][kk],targ_ar[jj][ind2[1:9]]] .= 2
			end
		
		end

		inh_nrn[inhib_ar[jj]] .= jj
	    
	end

	
	return Nrns, syn, inhib_ar, targ_ar, inhib, inh_nrn;
end

# ╔═╡ b47ed6fb-82dc-4a1c-98bf-870089d2c9e9
begin

nblocks=1
block_ar = Vector{Matrix{Float64}}(undef,nblocks)
inhib_ar = Vector{Vector{Int64}}(undef,nblocks)
targ_ar  = Vector{Vector{Int64}}(undef,nblocks)
	for ii = 1:nblocks

		block_ar[ii], inhib_ar[ii], targ_ar[ii] = cb_adj_gen(20,6);

		#push!(block_ar,block);
		#push!(inhib_ar,inh);
		#push!(targ_ar,targ);
	
	end

	Nrns, syn, inhib_ar, targ_ar, inhib, inh_nrn = connect_cb(block_ar,inhib_ar,targ_ar);


	N = 225
	S = 18
	I = 8


	Nrns = Nrns+N;
	for jj= 1:length(targ_ar)
	     targ_ar[jj] = targ_ar[jj] .+ N
		 inhib_ar[jj] = inhib_ar[jj] .+ N
	   
	end

	inhib = inhib .+ N
    inh_nrn = vcat(zeros(N),inh_nrn)
	inh_nrn = convert(Vector{Int64},inh_nrn)
	
	mat = zeros(Nrns,Nrns)
	mat[(N+1):end,(N+1):end] = syn
	syn=mat
	


	# connecting input to target cells
	wt = 2
	
	for ii = 1:length(targ_ar[1])
		
		mm = syn[targ_ar[1],1:N]
		ss = sum(mm,dims=1)
		rem = findall(x -> x<wt*I,ss[1,:])
	    ar=collect(1:length(rem))
		
		ar_sh = shuffle(ar)
		S_in = min(S,length(rem))
		input_nrns = rem[ar_sh[1:S_in]]
		syn[targ_ar[1][ii],input_nrns] .= wt
		
				
	end
	
	plot(Gray.(syn[N+1:end,:]/1))
end

# ╔═╡ f2f4b6b3-9098-4dcb-ac10-b838af07980a
begin
	ptrn = readdlm("Dist1.txt",',')
end;

# ╔═╡ 6d7ce7e5-65d3-4cf1-ab27-221cb07dd4a8
#simulation paremeters
begin
    simtime = 1000

	#input_pattern = P₂
	
    E_syn=zeros(1,Nrns);	
	E_syn[inhib] .=-70;

	G_syn= 3*ones(1,Nrns)#3;
	G_syn[inhib] .= 23;

	
	#I_in = zeros(Nrns);#10*ones(Nrns);
	#I_in[input_pattern] .=0.5;
    #I_in[inhib] .= 0.85;
       """
	   freq = zeros(Nrns);
       #freq[input_pattern] .= 5;
	   freq[inhib] .= 4;
	   freq[1:N] .= 4;

	   phase = pi*ones(Nrns);
	   phase[inhib] .= 0
       """
    τ = 5*ones(Nrns);
    τ[inhib] .= 70;

	freq1 = 4
    freq2 = 4
    phase_lag = 0
#	p = [syn,inh_nrnE_syn,G_syn,I_in,freq,phase,τ]
	
end

# ╔═╡ f18e3d9a-810f-4849-bbaa-4b6142dde625
begin
 asc_input1 = ascending_input(t,freq1,0)
 asc_input2 = ascending_input(t,freq2,phase_lag)
 input_ar = [asc_input1,asc_input2]
 for kk = 1:nblocks-2
 push!(input_ar,asc_input2)
 end
end;

# ╔═╡ c0943891-c172-432b-bb2f-59dedcebc07d
begin



if  rand()<=0.5 
	 println("1")
	 input_pattern = ptrn[:,rand(1:512)]
	else
	 input_pattern = ptrn[:,512+rand(1:512)]
		println("2")
end
	
	

	I_in = zeros(Nrns);#10*ones(Nrns);
	I_in[1:N] = 0.5*input_pattern;
    



	
end;

# ╔═╡ 15b613ff-5edb-49a7-b770-a2afcd361091
@parameters adj[1:Nrns*Nrns] = vec(syn)

# ╔═╡ f7f439ef-ba85-4023-b478-3f095fd9ff5b
function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix, input_ar=input_ar,inh_nrn = inh_nrn)
    syn_eqs= [ 0~sys[1].V - sys[1].V]
	        
    for ii = 1:length(sys)
       	
        presyn = findall(x-> x>0.0, adj_matrix[ii,:])
       # wts = adj_matrix[ii,presyn]		
		presyn_nrn = sys[presyn]
        postsyn_nrn = sys[ii]
		    
        if length(presyn)>0
					
		    ind = [i for i = 1:length(presyn)];
	      #  eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*wts[p],ind)-postsyn_nrn.Isyn]
           # push!(syn_eqs,eq[1])

			eq = [0 ~ sum(p-> (presyn_nrn[p].E_syn-postsyn_nrn.V)*presyn_nrn[p].G*adj[(presyn[p]-1)*Nrns + ii],ind)-postsyn_nrn.Isyn]
            push!(syn_eqs,eq[1])
			
		else
		    eq = [0~postsyn_nrn.Isyn];
		    push!(syn_eqs,eq[1]);
		 
		end

		if inh_nrn[ii]>0
            eq2 = [0 ~ postsyn_nrn.Iasc - input_ar[inh_nrn[ii]]];
			push!(syn_eqs,eq2[1])
		end
		
    end
    popfirst!(syn_eqs)
	
    @named synaptic_eqs = ODESystem(syn_eqs,t)
    
    sys_ode = [sys[ii] for ii = 1:length(sys)]

    @named synaptic_network = compose(synaptic_eqs, sys_ode)
    return structural_simplify(synaptic_network)   

end

# ╔═╡ 092502cf-4a04-4d70-b954-f3dfd2a6c9fa
begin

nrn_network=[]
	for ii = 1:Nrns
		if inh_nrn[ii]>0
nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
			
		else

nn = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],freq=freq1,phase=3*pi/2,τ=τ[ii])
		end
push!(nrn_network,nn)
	end


@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn, input_ar=input_ar, inh_nrn = inh_nrn)
	


end;

# ╔═╡ e1932634-20f9-4281-bbf9-6e910fc5dd8b
 prob = ODEProblem(syn_net, [], (0, simtime));

# ╔═╡ 65913ff6-f45b-4587-a12e-3dee5fc4d751
begin
# Accessing input pattern and adj matrix parameters from ODESystem
indexof(sym,syms) = findfirst(isequal(sym),syms)
global	cc=[]
for ii in 1:N
	#vvv = Sym{Real}(Symbol("nrn$ii","₊I_in"))
	 vvv = nrn_network[ii].I_in
global	cc=push!(cc,indexof(vvv,parameters(syn_net)))
end


global	dd=[]
	syn_v=vec(syn)
	
	in_con = findall(x-> x>0,syn_v)
for ii in in_con
	vvv = adj[ii]
global	dd=push!(dd,indexof(vvv,parameters(syn_net)))
end
	
end

# ╔═╡ c674133e-f9d2-4086-93e4-0f95b15e0f8a
parameters(syn_net)

# ╔═╡ 44835929-89ea-4f40-b2ca-7c9ddc14b3b7
(cc)

# ╔═╡ dcf47b84-02c9-4d46-9aee-3b3fb0a080c6
(in_con)

# ╔═╡ 28981872-896a-48b7-b466-a57979bfb338
sum(sign.(syn))

# ╔═╡ 0d4174a6-34b7-4162-9c1b-e4bd46415ffc
parameters(syn_net)[cc]

# ╔═╡ 12506766-1bde-4bea-8f97-dd03da463f26
Gray.(prob.p[cc])

# ╔═╡ 762ae21c-7047-42d2-a1ac-31ab2118b262
syn_net.ps

# ╔═╡ 778abea2-3899-46c2-8a23-96c26081a01b
#prob_new = remake(prob;p=prob_param)

# ╔═╡ 7a3e66cb-657d-420f-b474-bc7efc494665
soll = solve(prob,Vern7(),saveat = 0.1)#,saveat = 0.1,reltol=1e-4,abstol=1e-4);

# ╔═╡ 924e81c8-8479-4766-9e8e-ac3256a6bb08
begin
 ss = convert(Array,soll);
	VV=zeros(Nrns,length(soll.t));  V=zeros(Nrns,length(soll.t));
	
	for ii = 1:Nrns
		VV[ii,:] = ss[(((ii-1)*6)+1),1:end].+(ii-1)*200;
	   	V[ii,:] =  ss[(((ii-1)*6)+1),1:end];
	end
end

# ╔═╡ 41be95be-9cbc-475a-864a-894c6164a41c
plot(soll.t,[VV[1:100,:]'],legend=false,yticks=[],color = "blue",size = (1000,700))

# ╔═╡ 63cd9e70-a580-489d-9897-2fb127ef7c35
begin
pl1 = plot(soll.t,[VV[targ_ar[1][1:100],:]'],legend=false,yticks=[],color = "blue",size = (1000,700));
#pl1 = plot(sol.t,[VV[48+1:48+48,:]'],legend=false,yticks=[])

#plot!(pl1,sol.t,[VV[inhib_ar[3][1:end],:]'],legend=false,yticks=[],color = "red",ylabel = "single neuron \n activity")
end

# ╔═╡ a774461c-474e-42eb-9d6d-4ef642713ee3
length(findpeaks(V[targ_ar[1][37],:],soll.t,min_height=0.0))

# ╔═╡ 2e599fbb-7c4f-42e7-8e13-1c32758f41a7
length(V[targ_ar[1][37],:])

# ╔═╡ 58d67af0-a0ed-4489-b3b1-ddcc8ee0198d
maximum(V[targ_ar[1][37,:]])

# ╔═╡ dc4453d1-38ec-49a8-84b9-aa07cc01292f
syn2=readdlm("syn_mat_new.txt",',');

# ╔═╡ 27ffe2b9-1f4a-4cf6-a4c7-5efadf0ca72b
begin
#global adj=zeros(Nrns,Nrns)
#
#adj = copy(syn2)
#adj = convert(Matrix{Float64},adj)
#global loop = 1
end

# ╔═╡ b01c73fd-75bb-4cef-9f4d-5c8cbe6a0999
#learning process
begin

learning=0;

global loop=80

#	@show typeof(adj)
if learning ==1	





#while loop <= 10

	@info loop
	
if  rand()<=0.5 
	 input_pattern2 = ptrn[:,rand(1:512)]
	else
	 input_pattern2 = ptrn[:,512+rand(1:512)]
end
	
	

	I_in2 = zeros(Nrns);#10*ones(Nrns);
	I_in2[1:N] = 0.5*input_pattern2;
    
  
 #nrn_network=[]
for ii = 1:N
		
global nrn_network[ii] = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in2[ii],freq=freq1,phase=3*pi/2,τ=τ[ii])
		
end


@named syn_net2 = synaptic_network(sys=nrn_network,adj_matrix=adj, input_ar=input_ar, inh_nrn = inh_nrn)
	
       
prob2 = ODEProblem(syn_net2, [], (0, simtime));

	
	
solt = solve(prob,Vern7(),saveat = 0.1)


sol = convert(Array,solt);


    for ii = 1:length(targ_ar)
		
	  for kk = 1:length(targ_ar[ii])
		  
		#  @info kk
        v = sol[(((targ_ar[ii][kk]-1)*6)+1),1:end]
		if maximum(v)>0  
	    tt = findpeaks(v,solt.t,min_height=0.0)
		#if length(tt)>0

			if ii == 1
				preblock = collect(1:N)
			else
				preblock = targ_ar[ii-1]
				
			end
			
			source = adj[targ_ar[ii][kk],preblock]
			#act = sign.(I_in[1:N])
			#act_in = act.*pres
			source_con = findall(x-> x>0, source)
			pres = preblock[source_con]
			
			for jj = 1:length(pres)
				pre_v = sol[(((pres[jj]-1)*6)+1),1:end]
			if (adj[targ_ar[ii][kk],pres[jj]]<10) && (maximum(pre_v)>0) 
				pre_tt = findpeaks(pre_v,solt.t,min_height=0.0)
			  
				   adj[targ_ar[ii][kk],pres[jj]] = adj[targ_ar[ii][kk],pres[jj]] + 0.05*length(tt)*length(pre_tt)/20
				  #GC.gc()
				  #adj = convert(Matrix{Float64},adj)
			  end
			end

			
         #end

	 
	  end
	end
     end
#global	loop=loop+1
#end

end

end

# ╔═╡ 5bb4acd9-2db3-4b62-9617-66514ce8d8f1
begin

	global adj3=zeros(Nrns,Nrns)

   adj3 = copy(syn)
   adj3 = convert(Matrix{Float64},adj3)
	
	global str_in = [[Vector{Float64}(undef, length(targ_ar[1])) for _ = 1:2] for _ = 1:length(targ_ar)] 
	
#	global str_in = Array{Float64,3}

	for ii = 1:length(targ_ar)
	str_in[ii][1][:] = 0.1*rand(length(targ_ar[1]))
 	str_in[ii][2][:] = 0.1*rand(length(targ_ar[1]))
	end
	
	block_wt = ones(length(targ_ar))
	trial = zeros(500)
	trial_dec = zeros(500)
	#global tan_input = 100*ones(1);	

	#adj3 = readdlm("syn_mat.txt",',')
	#str_wt = readdlm("str_input.txt",',')

	#str_in[1][1][:] = str_wt[1,:]
	#str_in[1][2][:] = str_wt[2,:]
	
end

# ╔═╡ f1642d2d-a8c0-4936-96b6-13d48be60da7
begin
adj_vec=vec(adj3)
con_ind = findall(x-> x>0,adj_vec)
prob_param=copy(prob.p)
prob_param[dd] = adj_vec[con_ind]
prob_new = remake(prob;p=prob_param)
end

# ╔═╡ 58d59ecf-d09d-4dfa-9bd3-4fb6b5300bc2
begin
   adj_rec = zeros(120*100,225)
   CS1 = zeros(100,100)
   CS2 = zeros(100,100)
  
	
end

# ╔═╡ a39c90d8-c006-48f7-8191-cfb32855f52a
#Cortico-Striatal learning process
begin

str_learning=0;

spk_count1 = zeros(500,100)
spk_count2 = zeros(500,100)
catg_count = zeros(2)
I_in_arr3 = zeros(500,225)
#global loop3=1

global tan_input1 = 50;	
global tan_input2 = 50;	
	
	
des_12 = zeros(100)
#	@show typeof(adj)
if str_learning ==1	
global cnt = 0
for loop3 =1:500

	@info loop3

	#if mod(loop3,3)==1
	#	global cnt=cnt+1
	#	adj_rec[((cnt-1)*120+1):cnt*120,:] = adj3[226:end,1:225]
	#	CS1[cnt,:] = str_in[1][1][:]
	#	CS2[cnt,:] = str_in[1][2][:]
		
	#end
	
if  rand()<=0.5 
	 input_pattern3 = ptrn[:,rand(1:512)]
	 catg =1
	 catg_count[1] = catg_count[1] +1
	else
	 input_pattern3 = ptrn[:,512+rand(1:512)]
	 catg =2
	 catg_count[2] = catg_count[2] +1
end

	
#	I_in_arr3[loop3,:] = input_pattern3


	

	I_in3 = zeros(Nrns);#10*ones(Nrns);
	I_in3[1:N] = 0.5*input_pattern3;


prob_param=copy(prob.p)

adj_vec = vec(adj3)
conn_ind = findall(x-> x>0,adj_vec)

prob_param[cc] = I_in3[1:N]
prob_param[dd] = adj_vec[conn_ind]
  
 #nrn_network=[]
#for ii = 1:N
		
#global nrn_network[ii] = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in3[ii],freq=freq1,phase=3*pi/2,τ=τ[ii])
		
#end


#@named syn_net3 = synaptic_network(sys=nrn_network,adj_matrix=adj3, input_ar=input_ar, inh_nrn = inh_nrn)
	
       
prob3 = remake(prob;p=prob_param);

	
	
solt3 = solve(prob3,Vern7(),saveat = 0.1)


sol3 = convert(Array,solt3);

decision_ar = zeros(length(targ_ar))
spks = Vector{Vector{Float64}}(undef,length(targ_ar))
    for ii = 1:length(targ_ar)

		#decision making
		#catg1=zeros(length(targ_ar[ii])) # weighted input
		#catg1=zeros(length(targ_ar[ii]))
		spks[ii] = zeros(length(targ_ar[ii]))
		for ll = 1:length(targ_ar[ii])
		
		    v = sol3[(((targ_ar[ii][ll]-1)*6)+1),1:end]
		    if maximum(v)> 0 
		       tt = findpeaks(v,solt3.t,min_height=0.0)
		       spks[ii][ll] = length(tt)


				 if catg ==1
			       indx = convert(Int64,catg_count[1])
			       spk_count1[indx,ll] = length(tt)
		         else
			        indx = convert(Int64,catg_count[2])
			        spk_count2[indx,ll] = length(tt)
		         end
				
		
			end
		
		
		end

		
	#	act_syn1 = findall(x-> x>0.8,str_in[ii][1][:])
	#	act_syn2 = findall(x-> x>0.8,str_in[ii][2][:])

 #  global	tan_input1 = 50*0.9^(sum(spks[ii] .* str_in[ii][1][1:end]))*0.99^(minimum([loop3,30]));
 #  global	tan_input2 = 50*0.9^(sum(spks[ii] .* str_in[ii][2][1:end]))*0.99^(minimum([loop3,30]));

if loop3<=30
corr_dop = length(findall(x-> x>0,trial[1:loop3]))
incorr_nodop = loop3-corr_dop
net_dop = (corr_dop-incorr_nodop)
else
corr_dop = length(findall(x-> x>0,trial[(loop3-30):loop3-1]))
incorr_nodop = 30-corr_dop
net_dop = (corr_dop-incorr_nodop)
end
		
   global	tan_input1 = 100*0.99^(sum(spks[ii] .* str_in[ii][1][1:end]))*0.9^(maximum([net_dop,0]));
   global	tan_input2 = 100*0.99^(sum(spks[ii] .* str_in[ii][2][1:end]))*0.9^(maximum([net_dop,0]));
		
	des1 = sum(spks[ii] .* str_in[ii][1][1:end]) +  rand(Poisson(tan_input1))
	des2 = sum(spks[ii] .* str_in[ii][2][1:end]) +  rand(Poisson(tan_input2))

	#des1 = sum(spks[ii][act_syn1]) +  tan_input*rand()
	#des2 = sum(spks[ii][act_syn2]) +  tan_input*rand()
		#des_12[ii] = des1-des2;
		println([sum(spks[ii] .* str_in[ii][1][1:end]),sum(spks[ii] .* str_in[ii][2][1:end]), des1, des2])
		#println(des2)
		
		if des1>=des2   #decision from each block
			decision_ar[ii] =1
		else
			decision_ar[ii] =2
		end
		
		#if rand()<=0.5
		#	decision_ar[ii]=1
		#else
		#	decision_ar[ii]=2
		#end
		
		
		
		@info decision_ar[ii]	
		
	end
	act1 = findall(x-> x==1, decision_ar)
	act2 = findall(x-> x==2, decision_ar)

	#grand decision
	if sum(block_wt[act1]) >= sum(block_wt[act2]) #weighing decisions from all blocks
	   decision = 1
	   trial_dec[loop3]=1
		#push!(trial_dec,1)
	else
	   decision = 2
	   trial_dec[loop3]=2
	   #push!(trial_dec,2)	
	end
		
 #feedback driven learing
  for ii = 1:length(targ_ar)	

		 #cortical learning
	   for kk = 1:length(targ_ar[ii])
		 if decision == catg 
		#  @info kk
        #v = sol3[(((targ_ar[ii][kk]-1)*6)+1),1:end]
		if spks[ii][kk]>0  

			
	    #tt = findpeaks(v,solt3.t,min_height=0.0)
		#if length(tt)>0

		

			if ii == 1
				preblock = collect(1:N)
			else
				preblock = targ_ar[ii-1]
				
			end
			
			source = adj3[targ_ar[ii][kk],preblock]
			#act = sign.(I_in[1:N])
			#act_in = act.*pres
			source_con = findall(x-> x>0, source)
			pres = preblock[source_con]
			
			for jj = 1:length(pres)
				pre_v = sol3[(((pres[jj]-1)*6)+1),1:end]
			if (adj3[targ_ar[ii][kk],pres[jj]]<7) && (maximum(pre_v)>0) 
				pre_tt = findpeaks(pre_v,solt3.t,min_height=0.0)
			  
				   adj3[targ_ar[ii][kk],pres[jj]] = adj3[targ_ar[ii][kk],pres[jj]] + 0.05*spks[ii][kk]*length(pre_tt)/20
				  #GC.gc()
				  #adj = convert(Matrix{Float64},adj)
			  end
			end

			
         #end

	 
	    end

		

			 
	 end
   end

# feedback driven striatal leaning		 

   for jj = 1:length(targ_ar[ii])

	   if spks[ii][jj]>0

          if decision == catg #dopamine release

			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = minimum([str_in[ii][1][jj] + (0.1*spks[ii][jj]/20)*tan_input1/100,1])
				#  str_in[ii][2][jj] = maximum([str_in[ii][2][jj] - (0.1*spks[ii][jj]/20)*tan_input2/100,0])
			#	  str_in[ii][1][jj] = minimum([str_in[ii][1][jj] + (0.2*spks[ii][jj]/16),10])
			#	  str_in[ii][2][jj] = maximum([str_in[ii][2][jj] - (0.3*spks[ii][jj]/16),0])
			  else
				  str_in[ii][2][jj] = minimum([str_in[ii][2][jj] + (0.1*spks[ii][jj]/20)*tan_input2/100,1])
			#	  str_in[ii][1][jj] = maximum([str_in[ii][1][jj] - (0.1*spks[ii][jj]/20)*tan_input1/100,0])
	 		      
			#	  str_in[ii][2][jj] = minimum([str_in[ii][2][jj] + (0.2*spks[ii][jj]/16),10])
			#	  str_in[ii][1][jj] = maximum([str_in[ii][1][jj] - (0.3*spks[ii][jj]/16),0])
			  end
	   
		  else #dopamine not released

			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = maximum([str_in[ii][1][jj] - (0.2*spks[ii][jj]/20)*tan_input1/100,0])
			#	   str_in[ii][1][jj] = maximum([str_in[ii][1][jj] - (0.1*spks[ii][jj]/16),0])
			  else
				  str_in[ii][2][jj] = maximum([str_in[ii][2][jj] - (0.2*spks[ii][jj]/20)*tan_input2/100,0])
			#	   str_in[ii][2][jj] = maximum([str_in[ii][2][jj] - (0.1*spks[ii][jj]/16),0])
			  end


		  end
#no spiking
		else
              str_in[ii][1][jj] = str_in[ii][1][jj]*0.95
			  str_in[ii][2][jj] = str_in[ii][2][jj]*0.95
			
	   
	   end
      
   end

	if (decision == catg) 

		#println("Correct!")
		@info "Correct!"
		trial[loop3] = 1
		#push!(trial,1)

		if decision_ar[ii] == decision
			block_wt[ii] = block_wt[ii] + 0.05
		end

	else
        
		#println("Incorrect!")
        @info "Incorrect!"
		#push!(trial,0)
		if decision_ar[ii] == decision
			block_wt[ii] = maximum([(block_wt[ii] - 0.05),0])
		end
		

	end

	  
end

# tan_input[1] = tan_input[1]*0.95
end
end
end

# ╔═╡ 50b9da1a-be63-447c-bcee-1addfc2f6dc5
begin 
str_in_new = zeros(2,100);
str_in_new[1,:] = str_in[1][1][:]
str_in_new[2,:] = str_in[1][2][:]
"""
open("adj_rec.txt", "w") do io
    writedlm(io, adj_rec, ",")
end

open("CS1.txt", "w") do io
    writedlm(io, CS1, ",")
end
open("CS2.txt", "w") do io
    writedlm(io, CS2, ",")
end

open("syn_mat.txt", "w") do io
    writedlm(io, adj3, ",")
end

open("str_input.txt", "w") do io
    writedlm(io, str_in_new, ",")
end	
"""	
end
	          

# ╔═╡ db577a11-69bc-4195-89d3-5a3d4eef1cd0
0.99^100*100

# ╔═╡ dcf970a2-5413-43ed-a960-d39b6305778d
tan_input1

# ╔═╡ fafb048a-0d53-4e24-a629-9c89b00a3617
Gray.(trial[1:500])

# ╔═╡ ade2ba5c-c5d8-40ff-8f8f-6235c41cc6d8
trial

# ╔═╡ 8a50d29b-2c08-49f2-8ab3-de8ed613c1e2
length(trial)

# ╔═╡ 5e19bfd2-96d1-4091-9b3a-d84907ff2877
begin
"""
	open("performance4_high_distort.txt", "w") do io
          writedlm(io, trial, ",")
	end

	open("decision4_high_distort.txt", "w") do io
          writedlm(io, trial_dec, ",")
	end
"""
		  end

# ╔═╡ d5cbb034-43fc-4d24-954d-e8371c2be95b
trial_dec

# ╔═╡ f25c4843-0a4c-41fa-a0c3-1f2fd66f84ea
length(findall(x-> x==2,trial_dec))

# ╔═╡ ffe5aa9b-62bc-43d0-987e-4126fbb8db58
Gray.(trial_dec[1:500]./2)

# ╔═╡ 75c240e3-2195-4fd0-822b-6d8de34ef4cb
sum(trial[1:500])

# ╔═╡ c3660abd-8f8e-4c83-b8fe-bb553377fad0
maximum(adj3)

# ╔═╡ ae391335-8f23-4447-8676-ff3cc3a6f6be
maximum(str_in[1][1])

# ╔═╡ 8a1aceed-f5a8-43d7-a20e-bf3551b9dc27
maximum(str_in[1][2])

# ╔═╡ eb0cce33-5183-4104-b4fc-9a6b80255229
(Gray.((str_in[1][1]/maximum(str_in[1][1]))))

# ╔═╡ cdd10ed2-8bfa-4fab-92c7-edd82fd7065d
(Gray.((str_in[1][2]/maximum(str_in[1][2]))))

# ╔═╡ 2ab2bdde-6a14-470f-9f40-dcb10e82d7cb
catg_count

# ╔═╡ 7b8166e7-deef-4b56-8539-31e872d2e864
maximum(spk_count2)

# ╔═╡ d4d6ee5f-a766-48b5-accf-9cfce365e1fa
Gray.(spk_count1[1:128,:]./maximum(spk_count1))

# ╔═╡ 53b49ecf-e499-4dbc-ae2b-6b7c4e983ac3
Gray.(spk_count2[1:128,:]./maximum(spk_count2))

# ╔═╡ d20509e8-99d4-4c58-829d-dc4ac10997ab
plot(Gray.(1 .+sign.(adj3[N+1:end,:] .-7)))

# ╔═╡ e521f76a-a2a9-483f-ab50-925992f6e337
#testing process
begin

testing=0;

#loop=15
global adj2=zeros(Nrns,Nrns)

N_samples = 50
adj2 = copy(syn)
adj2 = convert(Matrix{Float64},adj2)

I_in_arr = zeros(N_samples,225)
activity_arr = Vector{Matrix{Float64}}(undef,nblocks)
#	@show typeof(adj)
if testing ==1	

activity = Vector{Vector{Float64}}(undef,nblocks)
activity2 = Vector{Vector{Float64}}(undef,nblocks)

for ii = 1:nblocks
        activity[ii] = zeros(length(targ_ar[ii]))
		activity2[ii] = zeros(length(targ_ar[ii]))
        activity_arr[ii] = zeros(N_samples,length(targ_ar[ii]))
	
end

for loop = 1:N_samples

	@info loop
	
if  loop<=N_samples/2 
	 input_pattern2 = ptrn[:,rand(1:512)]
else
	 input_pattern2 = ptrn[:,512+rand(1:512)]
end
	
I_in_arr[loop,:] = input_pattern2

	I_in2 = zeros(Nrns);#10*ones(Nrns);
	I_in2[1:N] = 0.5*input_pattern2;
    
 if loop >0 
 #nrn_network=[]
#for ii = 1:N
		
#global nrn_network[ii] = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in2[ii],freq=freq1,phase=3*pi/2,τ=τ[ii])
		
#end


#@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=adj2, input_ar=input_ar, inh_nrn = inh_nrn)
	
prob_param2=copy(prob.p)

adj_vec2 = vec(adj2)
con_ind2 = findall(x-> x>0,adj_vec2)

prob_param2[cc] = I_in2[1:N]
prob_param2[dd] = adj_vec2[con_ind2]      
	
 
prob2 = remake(prob;p=prob_param2); 
	
solt = solve(prob2,Vern7(),saveat = 0.1)


sol = convert(Array,solt);


    for ii = 1:length(targ_ar)

		
		
	  for kk = 1:length(targ_ar[ii])
		  
		#  @info kk
        v = sol[(((targ_ar[ii][kk]-1)*6)+1),1:end]
		if maximum(v)>0  
	      tt = findpeaks(v,solt.t,min_height=0.0)
		
            if loop<=N_samples/2
			    activity[ii][kk] = activity[ii][kk] + length(tt)
			else
				activity2[ii][kk] = activity2[ii][kk] + length(tt)
			end
			activity_arr[ii][loop,kk] = length(tt)
			
			  
		
			  
		end

			
         

	 
	  
	 end
    end
end
end
end
end

# ╔═╡ 959b1cbe-a4d4-4e77-b84a-040c73d81c96
Gray.(I_in_arr)

# ╔═╡ 600ed4d7-6bb2-4257-99e3-14eb0a49a3c5
Gray.(activity_arr[1]./maximum(activity_arr[1]))

# ╔═╡ 1dc1ea46-93af-49c2-8e6b-ad836ad8fc4f
activity_arr[1]

# ╔═╡ 0597d29e-d99c-4985-a0b8-02cd9b70ba13
activity2[1]

# ╔═╡ 7c431ead-fa71-427b-987e-485fa26d6353
begin
b1=bar(collect(1:100),activity[1],legend=false,ylims=(0,300));
b2=bar(collect(1:100),activity2[1],legend=false,ylims=(0,300),color="red");
#b3=bar(collect(1:100),activity[2],legend=false,ylims=(0,200));
#b4=bar(collect(1:100),activity2[2],legend=false,ylims=(0,200),color="red");
#b5=bar(collect(1:100),activity[3],legend=false,ylims=(0,200));
#b6=bar(collect(1:100),activity2[3],legend=false,ylims=(0,200),color="red");
#bar_cat_a = plot(b1,b3,b5,b2,b4,b6,layout=(2,3),color="blue",size = (1700,700))
bar_cat_a = plot(b1,b2,layout=(2,1),color="blue",size = (1700,700))
end

# ╔═╡ f26f8706-7ec2-4f63-96f7-5c50200a6e02
function angle_div(ar1,ar2)

	acos((ar1'*ar2)/sqrt((ar1'*ar1)*(ar2'*ar2)+0.00001))/(pi/2)
end

# ╔═╡ 33658a24-e2ed-407c-83b5-7a73b9827839
angle_div(str_in[1][1],str_in[1][2])

# ╔═╡ fe06afd1-357a-4cce-b4b1-903ee5402326
angle_div(activity[1],activity2[1])

# ╔═╡ 5578d83f-7598-40d8-b9c2-9785657e30e4
begin

	
input_div_mat = zeros(N_samples,N_samples)
output_div_mat = zeros(N_samples,N_samples*nblocks)

for ll = 1:nblocks	
for ii = 1:N_samples
	for jj = 1:N_samples
		input_div_mat[ii,jj] = angle_div(I_in_arr[ii,:],I_in_arr[jj,:])
		output_div_mat[ii,(jj + (ll-1)*N_samples)] = angle_div(activity_arr[ll][ii,:],activity_arr[ll][jj,:])
	end
end
end
end

# ╔═╡ c64f895f-bdc9-4098-a40f-8eea75c26299
maximum(adj3)

# ╔═╡ 02538234-cf7f-4e9e-948c-e5254a05d2d4
plot(Gray.(output_div_mat[:,:]))

# ╔═╡ fa0287e3-3a06-45c6-bc10-5cdb1d32b361
plot(Gray.(input_div_mat[:,:]))

# ╔═╡ 78f35071-14b2-4213-a9f8-9fc992c11cf0
begin
"""
	#open("syn_mat2.txt", "w") do io
     #     writedlm(io, syn, ",")
	#end

	#open("syn_mat_new2.txt", "w") do io
     #      writedlm(io, adj3, ",")
	#end


	open("input_div_mat_0_2.txt", "w") do io
           writedlm(io, input_div_mat, ",")
	end

	open("output_div_mat_0_high_distort2.txt", "w") do io
           writedlm(io, output_div_mat, ",")
	end

	
"""
end

# ╔═╡ Cell order:
# ╠═406f6214-cb40-11ec-037a-1325bda2f580
# ╠═1a01f8a2-d779-4b64-9401-3e746acdd6ab
# ╠═dbd16f92-8b0a-49c7-8bfd-1503967bdd9d
# ╠═c1ee3eed-5730-4ab1-a012-af6cce952024
# ╠═407ed4ff-9fec-4df8-a0ba-c264d1f7d2db
# ╠═5099a19c-0a25-4dc1-8442-ddf9ac56ef8f
# ╠═72112d41-4432-4233-9ab3-d9011674a3f8
# ╠═f7bb61b5-70f1-46ed-a8fd-bb26ca8fc32f
# ╠═8e6fcff1-3387-42b5-8d1f-8ba769adf6ca
# ╠═544f27bc-077a-488f-b9a4-8f4ca4cace4b
# ╠═7b070751-5d29-4f97-b4e0-899e35aa7041
# ╠═697586f1-0539-474f-99df-4106e39012ba
# ╠═4abaf4c3-14ac-4c82-a812-3fd4ee87e824
# ╠═0a803feb-3dd1-43ac-9afc-1b0afd19ce2d
# ╠═738fb9f1-81f3-4738-a8bd-407461c9586f
# ╠═ca25e5b5-9c81-461f-b014-54221ffd06c6
# ╠═e4b89f6a-21f0-42ca-950c-448afa5ec535
# ╠═61c5b42a-8723-4334-a3ba-8c8558b11284
# ╠═3be21966-09e5-46be-995c-c53e49d0a3c2
# ╠═ae38608c-2193-4439-b439-29fa7805c05f
# ╠═77b3c1e6-1ace-4a89-a364-ecd732a7b995
# ╠═b47ed6fb-82dc-4a1c-98bf-870089d2c9e9
# ╠═f2f4b6b3-9098-4dcb-ac10-b838af07980a
# ╠═6d7ce7e5-65d3-4cf1-ab27-221cb07dd4a8
# ╠═f18e3d9a-810f-4849-bbaa-4b6142dde625
# ╠═c0943891-c172-432b-bb2f-59dedcebc07d
# ╠═15b613ff-5edb-49a7-b770-a2afcd361091
# ╠═f7f439ef-ba85-4023-b478-3f095fd9ff5b
# ╠═092502cf-4a04-4d70-b954-f3dfd2a6c9fa
# ╠═e1932634-20f9-4281-bbf9-6e910fc5dd8b
# ╠═65913ff6-f45b-4587-a12e-3dee5fc4d751
# ╠═c674133e-f9d2-4086-93e4-0f95b15e0f8a
# ╠═44835929-89ea-4f40-b2ca-7c9ddc14b3b7
# ╠═dcf47b84-02c9-4d46-9aee-3b3fb0a080c6
# ╠═28981872-896a-48b7-b466-a57979bfb338
# ╠═0d4174a6-34b7-4162-9c1b-e4bd46415ffc
# ╠═12506766-1bde-4bea-8f97-dd03da463f26
# ╠═762ae21c-7047-42d2-a1ac-31ab2118b262
# ╠═778abea2-3899-46c2-8a23-96c26081a01b
# ╠═7a3e66cb-657d-420f-b474-bc7efc494665
# ╠═f1642d2d-a8c0-4936-96b6-13d48be60da7
# ╠═924e81c8-8479-4766-9e8e-ac3256a6bb08
# ╠═41be95be-9cbc-475a-864a-894c6164a41c
# ╠═63cd9e70-a580-489d-9897-2fb127ef7c35
# ╠═a774461c-474e-42eb-9d6d-4ef642713ee3
# ╠═2e599fbb-7c4f-42e7-8e13-1c32758f41a7
# ╠═58d67af0-a0ed-4489-b3b1-ddcc8ee0198d
# ╠═dc4453d1-38ec-49a8-84b9-aa07cc01292f
# ╟─27ffe2b9-1f4a-4cf6-a4c7-5efadf0ca72b
# ╟─b01c73fd-75bb-4cef-9f4d-5c8cbe6a0999
# ╠═5bb4acd9-2db3-4b62-9617-66514ce8d8f1
# ╠═58d59ecf-d09d-4dfa-9bd3-4fb6b5300bc2
# ╠═a39c90d8-c006-48f7-8191-cfb32855f52a
# ╠═50b9da1a-be63-447c-bcee-1addfc2f6dc5
# ╠═db577a11-69bc-4195-89d3-5a3d4eef1cd0
# ╠═dcf970a2-5413-43ed-a960-d39b6305778d
# ╠═fafb048a-0d53-4e24-a629-9c89b00a3617
# ╠═ade2ba5c-c5d8-40ff-8f8f-6235c41cc6d8
# ╠═8a50d29b-2c08-49f2-8ab3-de8ed613c1e2
# ╠═5e19bfd2-96d1-4091-9b3a-d84907ff2877
# ╠═d5cbb034-43fc-4d24-954d-e8371c2be95b
# ╠═f25c4843-0a4c-41fa-a0c3-1f2fd66f84ea
# ╠═ffe5aa9b-62bc-43d0-987e-4126fbb8db58
# ╠═75c240e3-2195-4fd0-822b-6d8de34ef4cb
# ╠═c3660abd-8f8e-4c83-b8fe-bb553377fad0
# ╠═ae391335-8f23-4447-8676-ff3cc3a6f6be
# ╠═8a1aceed-f5a8-43d7-a20e-bf3551b9dc27
# ╠═eb0cce33-5183-4104-b4fc-9a6b80255229
# ╠═cdd10ed2-8bfa-4fab-92c7-edd82fd7065d
# ╠═33658a24-e2ed-407c-83b5-7a73b9827839
# ╠═2ab2bdde-6a14-470f-9f40-dcb10e82d7cb
# ╠═7b8166e7-deef-4b56-8539-31e872d2e864
# ╠═d4d6ee5f-a766-48b5-accf-9cfce365e1fa
# ╠═53b49ecf-e499-4dbc-ae2b-6b7c4e983ac3
# ╠═d20509e8-99d4-4c58-829d-dc4ac10997ab
# ╠═e521f76a-a2a9-483f-ab50-925992f6e337
# ╠═959b1cbe-a4d4-4e77-b84a-040c73d81c96
# ╠═600ed4d7-6bb2-4257-99e3-14eb0a49a3c5
# ╠═1dc1ea46-93af-49c2-8e6b-ad836ad8fc4f
# ╠═0597d29e-d99c-4985-a0b8-02cd9b70ba13
# ╠═7c431ead-fa71-427b-987e-485fa26d6353
# ╠═fe06afd1-357a-4cce-b4b1-903ee5402326
# ╠═f26f8706-7ec2-4f63-96f7-5c50200a6e02
# ╠═5578d83f-7598-40d8-b9c2-9785657e30e4
# ╠═c64f895f-bdc9-4098-a40f-8eea75c26299
# ╠═02538234-cf7f-4e9e-948c-e5254a05d2d4
# ╠═fa0287e3-3a06-45c6-bc10-5cdb1d32b361
# ╠═78f35071-14b2-4213-a9f8-9fc992c11cf0
