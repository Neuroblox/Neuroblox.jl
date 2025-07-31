### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 406f6214-cb40-11ec-037a-1325bda2f580
import Pkg

# ╔═╡ 1a01f8a2-d779-4b64-9401-3e746acdd6ab
Pkg.activate(".")

# ╔═╡ abf532aa-d333-42fe-96de-b9ada89852e9
using DSP

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

# ╔═╡ 8872e8db-cb41-4068-8401-1721f07642c8
#str_in decay by 1% cort-cort max wt 5, thal_cort max wt 6, thal_cort outdeg = 8, CS syn = wt X 1, distort 2, CS decay = 0.99, thal_cort lr = 1/(200*500), str LR = 0.001

# ╔═╡ 64faf636-43c9-4aa9-8d78-e7aa4d91db50
#Pkg.add("PlutoUI")

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
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0 Gₛₜₚ(t)=0.0 
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ τ₃ = 2000 I_in = I_in freq=freq phase=phase
	
	
 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
stim_on(t) = 0.5*(1+sign(600-t))
#stim_on(t) = 0.5*(1+sign(1600-t))
	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in*stim_on(t)+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
		   D(Gₛₜₚ)~(-1/τ₃)*(Gₛₜₚ-0.0) + (z/5)*(0.5-Gₛₜₚ) #z/5, 0.5
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end
	

# ╔═╡ 233e2ddc-6148-459a-87ed-16646fea5316
function HH_neuron_wang_excit_thal(;name,E_syn=0.0,G_syn=2,I_in=0,freq=0,phase=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0 Gₛₜₚ(t)=0.0  
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ τ₃ = 2000 I_in = I_in freq=freq phase=phase
	
	
 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))

	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in+Isyn, 
	       D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
		   D(Gₛₜₚ)~(-1/τ₃)*Gₛₜₚ + (z/5)*(1-Gₛₜₚ)
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end
	

# ╔═╡ 3be21966-09e5-46be-995c-c53e49d0a3c2
function HH_neuron_wang_inhib(;name,E_syn=0.0,G_syn=2, I_in=0, τ=70)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Iasc(t) = 0.0 Isyn(t)=0.0 G(t)=0 z(t)=0 Gₛₜₚ(t)=0.0 
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = -0 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ τ₃ = 2000 I_in=I_in
	
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
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
	       D(Gₛₜₚ)~(-1/τ₃)*Gₛₜₚ + (z/5)*(1-Gₛₜₚ)
	      ]
	
	ODESystem(eqs,t,sts,ps;name=name)
end
	

# ╔═╡ c35348de-e55e-4c20-895e-f49a1bbeec4b
function HH_neuron_TAN_excit(;name,E_syn=0.0,G_syn=2,I_in=0,freq=0,phase=0,τ=10)
	sts = @variables V(t)=-65.00 n(t)=0.32 m(t)=0.05 h(t)=0.59 Isyn(t)=0.0 G(t)=0.0 z(t)=0.0 Gₛₜₚ(t)=0.0  
	
	ps = @parameters E_syn=E_syn G_Na = 52 G_K  = 20 G_L = 0.1 E_Na = 55 E_K = -90 E_L = -60 G_syn = G_syn V_shift = 10 V_range = 35 τ_syn = 10 τ₁ = 0.1 τ₂ = τ τ₃ = 2000 I_in = I_in freq=freq phase=phase
	
	
 αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
 βₙ(v) = 0.125*exp(-(v+44)/80)

	
 αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
 βₘ(v) = 4*exp(-(v+55)/18)
	 
 αₕ(v) = 0.07*exp(-(v+44)/20)
 βₕ(v) = 1/(1+exp(-(v+14)/10))	
	
	
ϕ = 5 
	
G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
#stim_on(t) = 0.5*(1+sign(600-t))
	
	eqs = [ 
		   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_in*(sin(t*freq*2*pi/1000)+1)+Isyn,
		   D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
	       D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
	       D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
           D(G)~(-1/τ₂)*G + z,
	       D(z)~(-1/τ₁)*z + G_asymp(V,G_syn),
		   D(Gₛₜₚ)~(-1/τ₃)*Gₛₜₚ + (z/5)*(1-Gₛₜₚ)
	      ]
	ODESystem(eqs,t,sts,ps;name=name)
end

# ╔═╡ 95653cf9-4db8-4cf9-b38b-fcb639c03bd7
function NextGenerationBlox(;name, C=30.0, Δ=1.0, η_0=5.0, v_syn=-10.0, alpha_inv=35.0, k=0.105)
        params = @parameters C=C Δ=Δ η_0=η_0 v_syn=v_syn alpha_inv=alpha_inv k=k
        sts    = @variables Z(t)=0.5 g(t)=1.6
        Z = ModelingToolkit.unwrap(Z)
        g = ModelingToolkit.unwrap(g)
        C, Δ, η_0, v_syn, alpha_inv, k = map(ModelingToolkit.unwrap, [C, Δ, η_0, v_syn, alpha_inv, k])
        eqs = [D(Z)~ (1/C)*(-im*((Z-1)^2)/2 + (((Z+1)^2)/2)*(-Δ + im*(η_0) + im*v_syn*g) - ((Z^2-1)/2)*g),
                    D(g) ~ alpha_inv*((k/(C*pi))*(1-abs(Z)^2)/(1+Z+conj(Z)+abs(Z)^2) - g)]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        #new(C, Δ, η_0, v_syn, alpha_inv, k, odesys.Z, odesys)
    end

# ╔═╡ ae38608c-2193-4439-b439-29fa7805c05f
#creates a single cortical block
#takes in number of wta units nblocks and size of each wta block, default 6
#gives out :
# syn : weight matrix of cortical block 
# inhib :  indices of feedback inhibitory neurons
# targ : indices of pyramidal neurons
# inhib_mod : indices of ascending gabaergic neurons

function cb_adj_gen(nblocks = 16, blocksize = 6)

	
	Nrns = blocksize*nblocks+2;

	#winner-take-all block
	mat = zeros(blocksize,blocksize);
	mat[end,1:end-1].=1#7;
	mat[1:end-1,end].=1;

	#disjointed blocks
	syn = zeros(Nrns,Nrns);
    for ii = 1:nblocks;
       syn[(ii-1)*blocksize+1:(ii*blocksize),(ii-1)*blocksize+1:(ii*blocksize)] = mat;
    end


	#feedback inhibitory neurons
	inhib = [kk*blocksize for kk = 1:nblocks]
	
    tot = [kk for kk=1:(Nrns-2)]
    targ = setdiff(tot,inhib); #target neurons

	#connecting wta blocks
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

	inhib_mod=Nrns-1;
	inhib_ff=Nrns;
	syn[inhib_ff,inhib_mod] = 0#1;
	syn[targ,inhib_ff] .=1;
	
	

	return syn, inhib, targ, inhib_mod, inhib_ff;
  
end

# ╔═╡ a42dcd5b-dc7b-47bf-8536-be6c21c2967b
#connects array of cortical blocks in series with hyper-geometric connections
#takes in:
#          block _ar: array of cortical block adj matrices
#          inhib_ar: array of arrays containing indices of inhibitory neurons in 
#                   respective blocks
#          targ_ar: array of arrays containing indices of pyramidal neurons in 
#                   respective blocks
#          inhib_mod : array of arrays containing indices of ascending neurons...
# 		   outdegree : number of outgoing synapses from each pyramidal neuron
#.         wt  :  synaptic weight of each 'inter-block' connection from one neuron to #                 another


# gives out:
#.          Nrns : total number of neurons (size of final adj matrix)
#.          syn : weight matrix of entire system 
#.          inhib_ar : array of array of indices for feedback inhibitory neurons 
#                      within respective blocks but re-indexed according to larger  
#.                     system 
#           targ_ar : array of array of indices for pyramidal neurons 
#                      within respective blocks but re-indexed according to larger  
#.                     system 
#.          inhib : inhib_ar concatenated into single array of system size Nrns
#           inh_nrn : array of size Nrns where indices corresponding to feedback 
#.                    inhibitory neurons have entries that indicate the cortical 
#                     block which they belong, rest every index has value 0  
#           inhib_mod_ar : array of array of indices for ascening inhibitory neurons 
#                      within respective blocks but re-indexed according to larger  
#.                     system 
#.          inhib_mod_nrn : array of size Nrns where indices corresponding to 
#.                          ascending inhibitory neurons have entries that indicate 
#                           the cortical block which they belong, rest every index 
#                           has value 0 
function connect_cb_hypergeometric(block_ar, inhib_ar, targ_ar, inhib_mod_ar,inhib_ff_ar,outdegree,wt)
	n_block = length(block_ar)
	l = zeros(n_block)
	inhib = inhib_ar[1]
	inhib_mod = inhib_mod_ar[1]
	inhib_ff=inhib_ff_ar[1]
	for ii = 1:n_block
		mat = block_ar[ii];
		l[ii] = length(mat[:,1])
		if ii>1
		    targ_ar[ii] = targ_ar[ii] .+ sum(l[1:(ii-1)])
			inhib_ar[ii] = inhib_ar[ii] .+ sum(l[1:(ii-1)])
			inhib_mod_ar[ii] = inhib_mod_ar[ii] + sum(l[1:(ii-1)])
			inhib_ff_ar[ii] = inhib_ff_ar[ii] + sum(l[1:(ii-1)])
			inhib = vcat(inhib,inhib_ar[ii])
			inhib_mod = vcat(inhib_mod,inhib_mod_ar[ii])
			inhib_ff = vcat(inhib_ff,inhib_ff_ar[ii])
			
		end
	end
	l = convert(Vector{Int64},l)
	Nrns = sum(l)
	syn = zeros(Nrns,Nrns)
	inh_nrn = zeros(Nrns)
	inh_mod_nrn = zeros(Nrns)
	inh_ff_nrn = zeros(Nrns)
	inh_nrn = convert(Vector{Int64},inh_nrn)
	inh_mod_nrn = convert(Vector{Int64},inh_mod_nrn)
	inh_ff_nrn = convert(Vector{Int64},inh_ff_nrn)
	
	for jj = 1:n_block
	    if jj==1
		 chk = 0
		else
		 chk = sum(l[1:(jj-1)])
		end
		
		syn[(chk+1):(chk+l[jj]),(chk+1):(chk+l[jj])] = block_ar[jj]

		if jj<n_block
           """
			lt1 = length(targ_ar[jj])
			lt2 = length(targ_ar[jj+1])
			for kk = 1:lt1
				ind2 = randperm(lt2)
				syn[targ_ar[jj+1][ind2[1:2]],targ_ar[jj][kk]] .= 1*3;
			end
			"""
            lt1 = length(targ_ar[jj])
			lt2 = length(targ_ar[jj+1])
			#wt = 3
            I = outdegree
			S = convert(Int64,ceil(I*lt1/lt2))

			for ii = 1:lt2
		
		       mm = syn[targ_ar[jj+1],targ_ar[jj]]
		       ss = sum(mm,dims=1)
		       rem = findall(x -> x<wt*I,ss[1,:])
	           ar=collect(1:length(rem))
		
		       ar_sh = shuffle(ar)
		       S_in = min(S,length(rem))
		       input_nrns = targ_ar[jj][rem[ar_sh[1:S_in]]]
		       syn[targ_ar[jj+1][ii],input_nrns] .= wt
		
			end
		
		end

		inh_nrn[inhib_ar[jj]] .= jj
		inh_mod_nrn[inhib_mod_ar[jj]] = jj
		inh_ff_nrn[inhib_ff_ar[jj]] = jj

	
	    
	end

	
	return Nrns, syn, inhib_ar, targ_ar, inhib, inh_nrn, inhib_mod_ar, inh_mod_nrn, inhib_ff_ar, inhib_ff, inh_ff_nrn;
end

# ╔═╡ b37c39ea-6746-48a9-b450-b3ea25530e7f
begin
function cort_str_loop_gen(Nrns,syn,targ_ar,N_str=25,N_GPi=25,N_thal=25,N_GPe=15,N_STN=15,N_TAN=1)

Nrns_tot = Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe+2*N_STN+N_TAN

syn_tot = zeros(Nrns_tot,Nrns_tot)

str_nrn = zeros(Nrns_tot)

GPi_nrn = zeros(Nrns_tot)

thal_nrn = zeros(Nrns_tot) 

GPe_nrn = zeros(Nrns_tot)

STN_nrn = zeros(Nrns_tot)

TAN_nrn = zeros(Nrns_tot)	
	
syn_tot[1:Nrns,1:Nrns] .= syn

str_ar  = Vector{Vector{Int64}}(undef,2)	
GPi_ar  = Vector{Vector{Int64}}(undef,2)
thal_ar  = Vector{Vector{Int64}}(undef,2)
GPe_ar = Vector{Vector{Int64}}(undef,2)
STN_ar = Vector{Vector{Int64}}(undef,2)
	

thal_ar_sh  = Vector{Vector{Int64}}(undef,2)		
cort_ar_sh  = Vector{Vector{Int64}}(undef,2)	
str_ar_sh  = Vector{Vector{Int64}}(undef,2)		
GPi_ar_sh  = Vector{Vector{Int64}}(undef,2)	
	
str_ar[1] = collect(Nrns+1:Nrns+N_str)	
str_ar[2] = collect(Nrns+N_str+1:Nrns+2*N_str)		
GPi_ar[1] = collect(Nrns+2*N_str+1:Nrns+2*N_str+N_GPi)
GPi_ar[2] = collect(Nrns+2*N_str+N_GPi+1:Nrns+2*N_str+2*N_GPi)	
thal_ar[1] = collect(Nrns+2*N_str+2*N_GPi+1:Nrns+2*N_str+2*N_GPi+N_thal)
thal_ar[2] = collect(Nrns+2*N_str+2*N_GPi+N_thal+1:Nrns+2*N_str+2*N_GPi+2*N_thal)
GPe_ar[1] = collect(Nrns+2*N_str+2*N_GPi+2*N_thal+1:Nrns+2*N_str+2*N_GPi+2*N_thal+N_GPe)
GPe_ar[2] = collect(Nrns+2*N_str+2*N_GPi+2*N_thal+N_GPe+1:Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe)
STN_ar[1]=(Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe+1:Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe+N_STN)
STN_ar[2]=(Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe+N_STN+1:Nrns+2*N_str+2*N_GPi+2*N_thal+2*N_GPe+2*N_STN)

TAN_ar = collect((Nrns_tot+1-N_TAN):Nrns_tot)
	
str_nrn[str_ar[1]] .= 1
str_nrn[str_ar[2]] .= 1	
GPi_nrn[GPi_ar[1]] .= 1
GPi_nrn[GPi_ar[2]] .= 1	
thal_nrn[thal_ar[1]] .= 1	
thal_nrn[thal_ar[2]] .= 1
GPe_nrn[GPe_ar[1]] .= 1
GPe_nrn[GPe_ar[2]] .= 1
STN_nrn[STN_ar[1]] .= 1
STN_nrn[STN_ar[2]] .= 1
TAN_nrn[TAN_ar] .= 1
# connect GPi to thal :

	GPi_outdeg = 10#1#4
	
	GPi_thal_wt = 0.16#0.2#1.6#2.5#3.5#1.5#6/25

	for ii = 1:length(thal_ar[1])

		#GPi_ar_sh = shuffle(GPi_ar)
		GPi_ar_sh[1] = shuffle(GPi_ar[1])
	    GPi_ar_sh[2] = shuffle(GPi_ar[2])
		
	#	syn_tot[thal_ar[ii],GPi_ar_sh[1:GPi_outdeg]] .= GPi_thal_wt
		
		syn_tot[thal_ar[1][ii],GPi_ar[1][ii]] = GPi_thal_wt
		syn_tot[thal_ar[2][ii],GPi_ar[2][ii]] = GPi_thal_wt

		#syn_tot[thal_ar[1][ii],GPi_ar_sh[1][1:GPi_outdeg]] .= GPi_thal_wt
		#syn_tot[thal_ar[2][ii],GPi_ar_sh[2][1:GPi_outdeg]] .= GPi_thal_wt 
		
	    
	
	end

#connect str to GPi :
    str_outdeg = 3#1#10
	
	str_GPi_wt = 4#3#1.3#7/25#6.5/25
	

      for jj = 1:length(GPi_ar[1])
		#str_ar_sh[1] = shuffle(str_ar[1])
		#str_ar_sh[2] = shuffle(str_ar[2])  
		  
		#syn_tot[GPi_ar[1][jj],str_ar_sh[1][1:str_outdeg]] .= str_GPi_wt  
		#syn_tot[GPi_ar[2][jj],str_ar_sh[2][1:str_outdeg]] .= str_GPi_wt    
		
		 syn_tot[GPi_ar[1][jj],str_ar[1][jj]] = str_GPi_wt 
		syn_tot[GPi_ar[2][jj],str_ar[2][jj]] = str_GPi_wt  
	  end	
# ========================================================
#connect thal to cort block

	ncb = length(targ_ar)
	thal_outdeg=8#20#8#4
    thal_cort_wt = 8#0.5#6#3#6#0.5#5#0.5#0.1*1#3
	   
	for kk = 1:length(targ_ar[ncb])
	      thal_ar_sh[1] = shuffle(thal_ar[1])
		  thal_ar_sh[2] = shuffle(thal_ar[2])
		   
		   syn_tot[targ_ar[ncb][kk],thal_ar_sh[1][1:thal_outdeg]] .= thal_cort_wt
		   syn_tot[targ_ar[ncb][kk],thal_ar_sh[2][1:thal_outdeg]] .= thal_cort_wt	  
    end
"""
	ncb = length(targ_ar)
	thal_outdeg=1
    thal_cort_wt = 2
	for kk = 1:length(thal_ar)
	       cort_ar_sh = shuffle(targ_ar[ncb])
		   
		   syn_tot[cort_ar_sh[1:thal_outdeg],thal_ar[kk]] .= thal_cort_wt	   
	end
"""	
#connect cort to str

	str_indeg = 4
    cort_str_wt = 0.06 
	for ll = 1:length(str_ar[1])
		cort_ar_sh[1] = shuffle(targ_ar[ncb])
		cort_ar_sh[2] = shuffle(targ_ar[ncb])
		syn_tot[str_ar[1][ll],cort_ar_sh[1][1:str_indeg]] .= cort_str_wt 
		syn_tot[str_ar[2][ll],cort_ar_sh[2][1:str_indeg]] .= cort_str_wt 
		
	end	

# =============================================
 #connect str to GPe	

	str_GPe_wt = 4#1.3#7/25#6.5/25
	

      for jj = 1:length(GPe_ar[1])
		str_ar_sh[1] = shuffle(str_ar[1])
		str_ar_sh[2] = shuffle(str_ar[2])  
		 
		syn_tot[GPe_ar[1][jj],str_ar_sh[1][jj]] = str_GPe_wt 
		syn_tot[GPe_ar[2][jj],str_ar_sh[2][jj]] = str_GPe_wt  
	  end	

 # connect GPe to GPi

	GPe_GPi_wt= 0.2

	   for jj = 1:length(GPe_ar[1])
		GPi_ar_sh[1] = shuffle(GPi_ar[1])  
		GPi_ar_sh[2] = shuffle(GPi_ar[2])     
		#syn_tot[GPi_ar[jj],str_ar_sh[1:str_outdeg]] .= str_GPi_wt  
		syn_tot[GPi_ar_sh[1][jj],GPe_ar[1][jj]] = GPe_GPi_wt 
		syn_tot[GPi_ar_sh[2][jj],GPe_ar[2][jj]] = GPe_GPi_wt  
	   end	

# connect GPe to STN
  
    GPe_outdeg =  1#4
	
	GPe_STN_wt = 3.5#1.5#6/25

	for ii = 1:length(STN_ar[1])

		#GPi_ar_sh = shuffle(GPi_ar)
		
	#	syn_tot[thal_ar[ii],GPi_ar_sh[1:GPi_outdeg]] .= GPi_thal_wt
		syn_tot[STN_ar[1][ii],GPe_ar[1][ii]] = GPe_STN_wt
		syn_tot[STN_ar[2][ii],GPe_ar[2][ii]] = GPe_STN_wt
	end

# connect  STN to GPi

	STN_GPi_wt= 0.1

   for ii = 1:length(GPe_ar[1])
        GPi_ar_sh[1] = shuffle(GPi_ar[1])  
		GPi_ar_sh[2] = shuffle(GPi_ar[2])

	    syn_tot[GPi_ar_sh[1][ii],STN_ar[1][ii]] = STN_GPi_wt
	    syn_tot[GPi_ar_sh[2][ii],STN_ar[2][ii]] = STN_GPi_wt
	   
   
   end

# connect thal to str

	thal_str_wt = 0#0.002#0.1

	 for ii = 1:3:length(thal_ar[1])
	    str_ar_sh[1] = shuffle(str_ar[1])
		str_ar_sh[2] = shuffle(str_ar[2]) 

		syn_tot[str_ar_sh[1][ii],thal_ar[1][ii]] = thal_str_wt 
		syn_tot[str_ar_sh[2][ii],thal_ar[2][ii]] = thal_str_wt 
	 
	 end

	
# connect TAN to str

	TAN_str_wt = 0.17

	syn_tot[str_ar[1][:],TAN_ar] .= TAN_str_wt
	syn_tot[str_ar[2][:],TAN_ar] .= TAN_str_wt
		
		
	
	
# ================================================

return Nrns_tot, syn_tot, str_nrn, GPi_nrn, thal_nrn, GPe_nrn, STN_nrn, TAN_nrn, str_ar, GPi_ar, thal_ar, GPe_ar, STN_ar, TAN_ar	
	
end
end

# ╔═╡ f2537041-c4d9-4f2f-be62-5c00a84f173d
#Adj matrix :create blocks and connencts them. First block is the input block recieving spatially patterned stimuli
begin

nblocks=2
block_ar = Vector{Matrix{Float64}}(undef,nblocks)
inhib_ar = Vector{Vector{Int64}}(undef,nblocks)
targ_ar  = Vector{Vector{Int64}}(undef,nblocks)
inhib_mod_ar = Vector{Int64}(undef,nblocks)
inhib_ff_ar = Vector{Int64}(undef,nblocks)	
str_ar  = Vector{Vector{Int64}}(undef,nblocks)	
GPi_ar  = Vector{Vector{Int64}}(undef,nblocks)
thal_ar  = Vector{Vector{Int64}}(undef,nblocks)	
	
block_ar[1], inhib_ar[1], targ_ar[1], inhib_mod_ar[1], inhib_ff_ar[1]= cb_adj_gen(45,6);	
	
	for ii = 2:nblocks

		block_ar[ii], inhib_ar[ii], targ_ar[ii], inhib_mod_ar[ii], inhib_ff_ar[ii]= cb_adj_gen(20,6);

	end
	
outdeg=8#4
wt=3#0.5#0.25#2
	
Nrns_cort, syn_cort, inhib_ar, targ_ar, inhib, inh_nrn, inhib_mod, inh_mod_nrn, inhib_ff_ar, inhib_ff, inh_ff_nrn = connect_cb_hypergeometric(block_ar,inhib_ar,targ_ar,inhib_mod_ar, inhib_ff_ar,outdeg,wt);

cort_block_num=2
N_str=25
N_GPi=25
N_thal=25
N_GPe=15
N_STN=15
N_TAN=1	
	
Nrns, syn, str_nrn, GPi_nrn, thal_nrn, GPe_nrn, STN_nrn, TAN_nrn, str_ar, GPi_ar, thal_ar, GPe_ar, STN_ar, TAN_ar = cort_str_loop_gen(Nrns_cort,syn_cort,targ_ar,N_str,N_GPi,N_thal,N_GPe,N_STN,N_TAN)	

inh_nrn = vcat(inh_nrn,zeros(2*N_str+2*N_GPi+2*N_thal+2*N_GPe+2*N_STN+N_TAN))	
inh_mod_nrn = vcat(inh_mod_nrn,zeros(2*N_str+2*N_GPi+2*N_thal+2*N_GPe+2*N_STN+N_TAN))
inh_ff_nrn = vcat(inh_ff_nrn,zeros(2*N_str+2*N_GPi+2*N_thal+2*N_GPe+2*N_STN+N_TAN))
	
inh_nrn = convert(Vector{Int64},inh_nrn)
inh_mod_nrn = convert(Vector{Int64},inh_mod_nrn)
inh_ff_nrn = convert(Vector{Int64},inh_ff_nrn)	
	
plot(Gray.(syn[targ_ar[2],:]/1))

end

# ╔═╡ f5294dac-d33d-4d61-b901-af8ac2b61dfe
plot(Gray.(sign.(syn./1)))

# ╔═╡ b47ed6fb-82dc-4a1c-98bf-870089d2c9e9
#Adj matrix :create blocks, connencts them and connec firs block with input axons
begin
"""
nblocks=1
block_ar = Vector{Matrix{Float64}}(undef,nblocks)
inhib_ar = Vector{Vector{Int64}}(undef,nblocks)
targ_ar  = Vector{Vector{Int64}}(undef,nblocks)
inhib_mod_ar = Vector{Int64}(undef,nblocks)
	
	for ii = 1:nblocks

		block_ar[ii], inhib_ar[ii], targ_ar[ii], inhib_mod_ar[ii] = cb_adj_gen(20,6);

		#push!(block_ar,block);
		#push!(inhib_ar,inh);
		#push!(targ_ar,targ);
	
	end

	Nrns, syn, inhib_ar, targ_ar, inhib, inh_nrn = connect_cb(block_ar,inhib_ar,targ_ar,inhib_mod_ar);


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
"""	
end

# ╔═╡ f2f4b6b3-9098-4dcb-ac10-b838af07980a
begin
	ptrn = readdlm("Dist4.txt",',')
end;

# ╔═╡ 88ebe172-46a3-4032-acf3-950e5d9ab7a6
inhib_mod[2]

# ╔═╡ dc575aaf-887e-40e0-9e19-235e16532735
inhib_ff

# ╔═╡ 6d7ce7e5-65d3-4cf1-ab27-221cb07dd4a8
#simulation paremeters
begin
    simtime = 1600#1000


	
    E_syn=zeros(1,Nrns);	
	E_syn[inhib] .=-70;
	E_syn[inhib_mod] .= -70;
	E_syn[inhib_ff] .= -70;
	E_syn[str_ar[1]] .= -70;
	E_syn[str_ar[2]] .= -70;
    E_syn[GPi_ar[1]] .= -70;
	E_syn[GPi_ar[2]] .= -70;
	E_syn[GPe_ar[1]] .= -70;
	E_syn[GPe_ar[2]] .= -70;
	#E_syn[STN_ar[1]] .= -70;
	#E_syn[STN_ar[2]] .= -70;
	

	G_syn=3*ones(1,Nrns);
	G_syn[inhib] .= 4#3#0.5*23;#25;
	G_syn[inhib_mod[1]] = 7.5#6#2*20;
	G_syn[inhib_mod[2]] = 7.5#6#2*20;
	  G_syn[inhib_ff[1]] = 3.5#1.3#3
	  G_syn[inhib_ff[2]] = 3.5#1.3#3
    G_syn[str_ar[1]] .= 1.2#0.3*10;
	G_syn[str_ar[2]] .= 1.2#0.3*10;
	G_syn[GPi_ar[1]] .= 8#0.3*10;
	G_syn[GPi_ar[2]] .= 8#0.3*10;
	G_syn[GPe_ar[1]] .= 0.3*10;
	G_syn[GPe_ar[2]] .= 0.3*10;
	G_syn[STN_ar[1]] .= 0.3*10;
	G_syn[STN_ar[2]] .= 0.3*10;
	
	τ = 5*ones(Nrns);
	τ[inhib] .= 70;
	τ[inhib_mod] .= 70;
	τ[inhib_ff] .= 70;
	τ[str_ar[1]] .= 70;
	τ[str_ar[2]] .= 70;
	τ[GPi_ar[1]] .= 70;
	τ[GPi_ar[2]] .= 70;
	τ[GPe_ar[1]] .= 70;
	τ[GPe_ar[2]] .= 70;
    #τ[STN_ar[1]] .= 70;
	#τ[STN_ar[2]] .= 70;
	
	freq1=16#16#16;
	freq2=16;
	freq3 = 12;
	freq_str=12;
	phase_lag=0;

	I_in = zeros(Nrns);
    I_in[GPi_ar[1]] = 4*ones(length(GPi_ar[1])) + 0.0*randn(length(GPi_ar[1]));#5
	I_in[GPi_ar[2]] = 4*ones(length(GPi_ar[2])) + 0.0*randn(length(GPi_ar[2]));#5
	#I_in[thal_ar[1]] .= (15/30)*30*ones(length(thal_ar[1])) + 0.0*randn(length(thal_ar[1]));#20 
	#I_in[thal_ar[2]] .= (15/30)*30*ones(length(thal_ar[2])) + 0.0*randn(length(thal_ar[2]));#20 

	I_in[thal_ar[1]] .= (3/30)*30*ones(length(thal_ar[1])) + 0.0*randn(length(thal_ar[1]));#7#20 
	I_in[thal_ar[2]] .= (3/30)*30*ones(length(thal_ar[2])) + 0.0*randn(length(thal_ar[2]));#7#20 

	I_in[GPe_ar[1]] = 2*ones(length(GPe_ar[1])) + 0.0*randn(length(GPe_ar[1]));
	I_in[GPe_ar[2]] = 2*ones(length(GPe_ar[2])) + 0.0*randn(length(GPe_ar[2]));
	I_in[STN_ar[1]] .= 0.5*(10/30)*30*ones(length(STN_ar[1])) + 0.0*randn(length(STN_ar[1]));#20 
	I_in[STN_ar[2]] .= 0.5*(10/30)*30*ones(length(STN_ar[2])) + 0.0*randn(length(STN_ar[2]));#20 

	I_in[TAN_ar] .= 1.5

	#I_in[inhib_ff] .= 4.5

	#I_in[str_ar[2]] .=-2
	
	
end

# ╔═╡ c0943891-c172-432b-bb2f-59dedcebc07d
#setting the input pattern for test run for stimulus response
begin

if  rand()<=1#0.5 
	 println("1")
	 input_pattern = ptrn[:,1]#rand(1:512)]
	else
	 input_pattern = ptrn[:,512+rand(1:512)]
		println("2")
end
	
	
	I_in[targ_ar[1]] = 14*input_pattern;#14

end;

# ╔═╡ 15b613ff-5edb-49a7-b770-a2afcd361091
begin
@parameters adj[1:Nrns*Nrns] = vec(syn)
@parameters phase_pfc=0
end	

# ╔═╡ f18e3d9a-810f-4849-bbaa-4b6142dde625
# setting modulation frequencies and phases
begin
 amp = 0.8	
 str_amp = 0.2*0
 asc_input1 = ascending_input(t,freq1,phase_pfc,amp)
 asc_input2 = ascending_input(t,freq2,phase_pfc,amp)
 str_input = ascending_input(t,freq3,0,str_amp)	
 input_ar = [asc_input1,asc_input2]
 for kk = 1:nblocks-2
 push!(input_ar,asc_input2)
 end
 push!(input_ar,str_input)	
end;

# ╔═╡ f7f439ef-ba85-4023-b478-3f095fd9ff5b
#constructs ODESystem for entire system
function synaptic_network(;name, sys=sys, adj_matrix=adj_matrix, input_ar=input_ar,inh_nrn = inh_nrn,inh_mod_nrn=inh_mod_nrn, inh_ff_nrn, str_nrn=str_nrn, GPi_nrn=GPi_nrn, thal_nrn=thal_nrn, GPe_nrn=GPe_nrn, STN_nrn=STN_nrn)
    syn_eqs= [ 0~sys[1].V - sys[1].V]

	thal=findall(x->x>0, thal_nrn)
	        
    for ii = 1:length(sys)
       	
        presyn = findall(x-> x>0.0, adj_matrix[ii,:])
		presyn_crt = setdiff(presyn,thal)
		presyn_thal= intersect(presyn,thal)
      	presyn_nrn_crt = sys[presyn_crt]
		presyn_nrn_thal = sys[presyn_thal]
        postsyn_nrn = sys[ii]

		if length(presyn_crt)>0
			ind_c = collect(1:length(presyn_crt))
			syn_input = sum(p-> (presyn_nrn_crt[p].E_syn-postsyn_nrn.V)*presyn_nrn_crt[p].G*adj[(presyn_crt[p]-1)*Nrns + ii],ind_c)
		else
			syn_input = 0
		end

		if length(presyn_thal)>0
			ind_t = collect(1:length(presyn_thal))
			syn_input_stp = sum(p-> (presyn_nrn_thal[p].E_syn-postsyn_nrn.V)*presyn_nrn_thal[p].G*postsyn_nrn.Gₛₜₚ*adj[(presyn_thal[p]-1)*Nrns + ii],ind_t)
		else
			syn_input_stp = 0
		end

		eq = [0 ~ syn_input + syn_input_stp - postsyn_nrn.Isyn]
		push!(syn_eqs,eq[1])
		
		if (inh_mod_nrn[ii]>0) 
            eq2 = [0 ~ postsyn_nrn.Iasc - input_ar[inh_mod_nrn[ii]]];
			push!(syn_eqs,eq2[1])
		end

		if (inh_ff_nrn[ii]>0) 
            eq2 = [0 ~ postsyn_nrn.Iasc - input_ar[inh_ff_nrn[ii]]];
			push!(syn_eqs,eq2[1])
		end

		if (inh_nrn[ii]>0)#||(inh_ff_nrn[ii]>0)
            eq2 = [0 ~ postsyn_nrn.Iasc];
			push!(syn_eqs,eq2[1])
		end

		if str_nrn[ii]>0
			eq2 = [0 ~ postsyn_nrn.Iasc - input_ar[end]];
			push!(syn_eqs,eq2[1])
		end

		if GPi_nrn[ii]>0
		    eq2 = [0 ~ postsyn_nrn.Iasc];
			push!(syn_eqs,eq2[1])
		end

		if GPe_nrn[ii]>0 
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

# ╔═╡ 092502cf-4a04-4d70-b954-f3dfd2a6c9fa
begin

nrn_network=[]
	for ii = 1:Nrns
		if (inh_nrn[ii]>0) || (inh_mod_nrn[ii]>0) 
nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
		
		elseif inh_ff_nrn[ii]>0
nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])			
			
		elseif (str_nrn[ii]>0) || (GPi_nrn[ii]>0)
nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])

		elseif (GPe_nrn[ii]>0) 
nn = HH_neuron_wang_inhib(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])	

		elseif thal_nrn[ii]	>0 || (STN_nrn[ii]>0)
nn = HH_neuron_wang_excit_thal(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
			
		elseif TAN_nrn[ii] > 0
nn = HH_neuron_TAN_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],freq=freq_str,τ=τ[ii])			
		
		else

nn = HH_neuron_wang_excit(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],τ=τ[ii])
		end
push!(nrn_network,nn)
	end

	#LC = NextGenerationBlox(name=Symbol("LC"),C=103, Δ=0.1, η_0=26.0, v_syn=-15, alpha_inv=0.1, k=10)


@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn, input_ar=input_ar, inh_nrn=inh_nrn, inh_mod_nrn=inh_mod_nrn, inh_ff_nrn=inh_ff_nrn,str_nrn=str_nrn, GPi_nrn=GPi_nrn, thal_nrn=thal_nrn, GPe_nrn=GPe_nrn, STN_nrn=STN_nrn)
	


end;

# ╔═╡ e1932634-20f9-4281-bbf9-6e910fc5dd8b
 prob = ODEProblem(syn_net, [], (0, simtime));

# ╔═╡ ec6d5982-6e80-48ed-9d49-edc7fb07888c
begin

	# Accessing input pattern and adj matrix parameters from ODESystem
indexof(sym,syms) = findfirst(isequal(sym),syms)
global	cc=[]
for ii in targ_ar[1]
	
	 vvv = nrn_network[ii].I_in
global	cc=push!(cc,indexof(vvv,parameters(syn_net)))
end

global str_ind1=[]
global str_ind2=[]

	for ii in str_ar[1]
		vvv = nrn_network[ii].I_in
	    global 	str_ind1=push!(str_ind1,indexof(vvv,parameters(syn_net)))
	end

	for ii in str_ar[2]
		vvv = nrn_network[ii].I_in
	    global 	str_ind2=push!(str_ind2,indexof(vvv,parameters(syn_net)))
	end

global tan_ind=[]
	for ii in TAN_ar
		vvv = nrn_network[ii].I_in
		global tan_ind=push!(tan_ind,indexof(vvv,parameters(syn_net)))
	end

 
global	dd=[]
	syn_v=vec(syn)
	
	in_con = findall(x-> x>0,syn_v)
for ii in in_con
	vvv = adj[ii]
global	dd=push!(dd,indexof(vvv,parameters(syn_net)))
end

global phase_ind = indexof(phase_pfc,parameters(syn_net))	


end

# ╔═╡ 9a3721f6-2b99-400e-af9d-c3969b57369a
begin

   adj0 = vec(syn)

	con_ind0 = findall(x-> x>0,vec(syn))
    prob_param0=copy(prob.p)

	prob_param0[dd] .= adj0[con_ind0]
	prob_param0[cc] .= 0*I_in[targ_ar[1]]	
	prob0 = remake(prob;p=prob_param0, tspan = (0,500))
	
	soll = solve(prob0,Vern7(),saveat = 0.1)#,saveat = 0.1,reltol=1e-4,abstol=1e-4);

	ss = convert(Array,soll);
	
	VV0=zeros(Nrns,length(soll.t));  V0=zeros(Nrns,length(soll.t));
	GG0 = zeros(Nrns,length(soll.t));
	for ii = 1:Nrns
		VV0[ii,:] = ss[(((ii-1)*7)+1),1:end].+(ii-1)*200;
	   	V0[ii,:] =  ss[(((ii-1)*7)+1),1:end];
		GG0[ii,:] = ss[(((ii-1)*7)+7),1:end]
	end
	
end

# ╔═╡ 9f7fe203-ac41-4502-9b41-7028bf9f21b3
begin
	"""
adj_vec_t=vec(adj3)
trl_set=100	
spt=Vector{Vector{Float32}}(undef,100*trl_set)	

	fir_patt=1

if fir_patt==1
	
for loop4 = 1:trl_set
@info loop4	
	
if loop4 in collect(1:25) #catgA correct
input_pattern6 = ptrn[:,rand(1:512)]
I_in6 = zeros(Nrns);
I_in6[targ_ar[1]] = 14*input_pattern5 #14;
strin1=0
strin2=-2
end
	
if loop4 in collect(26:50) #catgA incorrect
input_pattern6 = ptrn[:,rand(1:512)]
I_in6 = zeros(Nrns);
I_in6[targ_ar[1]] = 14*input_pattern5 #14;
strin1=-2
strin2=0
end

if loop4 in collect(51:75) #catgB incorrect
input_pattern6 = ptrn[:,512+rand(1:512)]
I_in6 = zeros(Nrns);
I_in6[targ_ar[1]] = 14*input_pattern5 #14;
strin1=-2
strin2=0
end	

if loop4 in collect(76:100) #catgB incorrect
input_pattern6 = ptrn[:,512+rand(1:512)]
I_in6 = zeros(Nrns);
I_in6[targ_ar[1]] = 14*input_pattern5 #14;
strin1=0
strin2=-2
end		
	



#adj_vec=vec(adj_rec[:,:,650])
con_ind = findall(x-> x>0,vec(syn))
prob_param_t=copy(prob.p)
prob_param_t[cc] = I_in6[targ_ar[1]]	
prob_param_t[dd] = adj_vec_t[con_ind]
prob_param_t[str_ind1] .= strin1	
prob_param_t[str_ind2] .= strin2	
prob_param_t[phase_ind] = rand()*2*pi
	


prob_new_t = remake(prob;p=prob_param_t, u0=initial_state, tspan = (0,1600))
soll3 = solve(prob_new_t,Vern7(),saveat = 0.1)
sol_ar = convert(Array,soll3);


	V_fp=zeros(Nrns,length(soll3.t));
	
	for ii = 1:Nrns
		
	   	V_fp[ii,:] =  sol_ar[(((ii-1)*7)+1),1:end];
		
	end




	
	
 for ii = 1:100	
	
   v=V_fp[targ_ar[2][ii],:];
   time=soll3.t/10;
	
	tt = findpeaks(v,time,min_height=0.0)/10
    tt=sort(tt)
	 
	 if length(tt)==0
	     tt = zeros(10)
	 end
	 
	 spt[(loop4-1)*100+ii] = tt
	 
 end


	


end

	open("spt_pattern13.txt", "w") do io
          writedlm(io, spt, ",")
	end

end
"""
end

# ╔═╡ 143ab0df-0fd1-4add-ae31-1b734eb86ef4
begin
	"""
	V_pfc_b = V[targ_ar[2][:],:]
	V_pfc_ = (round.(V_pfc_b .*100))./100
	svec = [ii*10+1 for ii = 0:1600]
    V_pfc = V_pfc_[:,:] 

	V_pfc_in_b = V[inhib_ar[2][:],:]
	V_pfc_in_ = (round.(V_pfc_in_b .*100))./100
    V_pfc_in = V_pfc_in_[:,:] 

	G_stp_b =  GG[targ_ar[2][:],:]
	G_stp_ = (round.(G_stp_b .*10000))./10000
	G_stp = G_stp_[:,svec];
	
	G_pfc_in_b = GGG[inhib_ar[2][:],:]
	G_pfc_in_ = (round.(G_pfc_in_b .*100))./100
	
	
    G_pfc_in = G_pfc_in_[:,:] 
	

	open("V_pfc_trial13.txt", "w") do io
          writedlm(io, V_pfc, ",")
	end 

	open("V_pfc_in_trial13.txt", "w") do io
          writedlm(io, V_pfc_in, ",")
	end 

	open("G_pfc_in_trial13.txt", "w") do io
          writedlm(io, G_pfc_in, ",")
	end 

	open("G_stp_trial13.txt", "w") do io
          writedlm(io, G_stp, ",")
	end 
    """
	
end

# ╔═╡ ed2e05a2-15eb-48d0-9cf7-5a0468a3f05d
sum(sign.(I_in[targ_ar[1]]))

# ╔═╡ 192e8ee4-aa06-412e-97ad-0197da60bc86
begin
	"""
open("cort_nrn_burst.txt", "w") do io
          writedlm(io,V[targ_ar[2][21],:], ",")
	end

	open("LC_input.txt", "w") do io
          writedlm(io,aa, ",")
	end

	open("ff_nrn_burst.txt", "w") do io
          writedlm(io,V[inhib_ff[2],:], ",")
	end
"""
end

# ╔═╡ 60cae459-fddc-435f-a07a-081c9aa54529


# ╔═╡ a110ce12-bde6-401d-8aff-2dfbd99e57b5
begin

	global adj3=zeros(Nrns,Nrns) 

   adj3 = copy(syn)
   adj3 = convert(Matrix{Float64},adj3)
	
	global str_in = [[Vector{Float64}(undef, length(targ_ar[2])) for _ = 1:2] for _ = 1:length(targ_ar)] 
	
#	global str_in = Array{Float64,3}
    
	for ii = 1:(length(targ_ar))
	str_in[ii][1][:] = 0.15*rand(length(targ_ar[2])) #0.04
 	str_in[ii][2][:] = 0.15*rand(length(targ_ar[2])) #0.04
	end

	for ll=1:length(targ_ar[2])

	#     adj3[str_ar[1],targ_ar[2][ll]] = sign.(adj3[str_ar[1],targ_ar[2][ll]])*0.6*(str_in[2][1][ll]+0.02)

     #    adj3[str_ar[2],targ_ar[2][ll]] = sign.(adj3[str_ar[2],targ_ar[2][ll]])*0.6*(str_in[2][2][ll]+0.02)

		 adj3[str_ar[1],targ_ar[2][ll]] = sign.(adj3[str_ar[1],targ_ar[2][ll]])*1*(str_in[2][1][ll]+0.0)

         adj3[str_ar[2],targ_ar[2][ll]] = sign.(adj3[str_ar[2],targ_ar[2][ll]])*1*(str_in[2][2][ll]+0.0)

		

		
	 
	end
	
	block_wt = ones(length(targ_ar))
	trial = zeros(700)
	trial_dec = zeros(700)

	tan_input_ar1=zeros(700)
    tan_input_ar2=zeros(700)
    spike_input_ar1=zeros(700)
    spike_input_ar2=zeros(700)
    des_ar1=zeros(700)
    des_ar2=zeros(700)
    dop_ar=zeros(700)	

	global cort_str_in=zeros(700)
	
# relates prediction error to plasticity rate
	function CS_plast(delD,T_ltd)
       
		       if delD>=0
				   return delD
			   else
				   if delD >= T_ltd
					   return delD*(-1/T_ltd)
				   else
					   return -1 + (delD-T_ltd)*(-1/(T_ltd+1))
				   end
			   end
		
	end

	function CS_plast2(delD)
      delD1 = delD.*(sign.(-delD)+1)/2;
      delD2 = delD.*(sign.(delD)+1)/2;
		       
     lr1 = 8*delD1.*(delD1+1);
     lr2 = 1*tanh.(delD2*2);
     lr=lr1+lr2;
		
     end

	function CS_plast3(delD)
     
         lr =  40*delD.*(delD+1)./(exp(2.5*(delD+1))+exp(-2.5*(delD+1))+2);
         return lr; 
		
    end

	pfc_avg = zeros(700,1600*10+1)
	str_avg = zeros(700,1600*10+1)

	pfc_LSS = zeros(700,1600*10+1)
	str_LSS = zeros(700,1600*10+1)
	
	#global tan_input = 100*ones(1);	

	#adj3 = readdlm("syn_mat.txt",',')
	#str_wt = readdlm("str_input.txt",',')

	#str_in[1][1][:] = str_wt[1,:]
	#str_in[1][2][:] = str_wt[2,:]
	
end

# ╔═╡ 58478f96-2575-473c-bf54-4f34dcb421cc
begin

adj_vec=vec(adj3)

input_pattern5 = ptrn[:,512+rand(1:512)]
I_in5 = zeros(Nrns);
I_in5[targ_ar[1]] = 14*input_pattern5 #14;

#adj_vec=vec(adj_rec[:,:,650])
con_ind = findall(x-> x>0,vec(syn))
prob_param=copy(prob.p)
prob_param[cc] = I_in5[targ_ar[1]]	
prob_param[dd] = adj_vec[con_ind]
prob_param[str_ind1] .= -2	
prob_param[str_ind2] .= 0	
rp = rand()*2*pi
prob_param[phase_ind] = rp
#prob_param[phase_ind] = 0#rand()*2*pi
	

global initial_state = ss[:,end]	
	g_ind=[i*7 for i= 1:Nrns]
	initial_state[g_ind] .= 0.0
prob_new = remake(prob;p=prob_param, u0=initial_state, tspan = (0,1600))

end

# ╔═╡ 225e1431-bf2a-4855-abd6-f0d826b5abe8
soll2 = solve(prob_new,Vern7(),saveat = 0.1)
#soll2 = solve(prob,Vern7(),saveat = 0.1)
#soll2=soll

# ╔═╡ 86b87ae5-01c9-4fc8-b37a-e5bf8b8d9d72
begin
 ss2 = convert(Array,soll2);
	VV=zeros(Nrns,length(soll2.t));  V=zeros(Nrns,length(soll2.t));
	GG = zeros(Nrns,length(soll2.t));
	GGG = zeros(Nrns,length(soll2.t));
	
	for ii = 1:Nrns
		VV[ii,:] = ss2[(((ii-1)*7)+1),1:end].+(ii-1)*200;
	   	V[ii,:] =  ss2[(((ii-1)*7)+1),1:end];
		GG[ii,:] = ss2[(((ii-1)*7)+7),1:end]
		GGG[ii,:] = ss2[(((ii-1)*7)+5),1:end]
	end

	V_spks=(sign.(V .+30) .+1) ./2

   average_VC = mean(V[targ_ar[1][:],:],dims=1);
   average_pfc = mean(V[targ_ar[2][:],:],dims=1);
   average_str1 = mean(V[str_ar[1],:],dims=1)	
   average_str2 = mean(V[str_ar[2],:],dims=1)		
   average_thal1 = mean(V[thal_ar[1],:],dims=1)	
	average_thal2 = mean(V[thal_ar[2],:],dims=1)	


	average_VC_ = mean(V_spks[targ_ar[1][:],:],dims=1);
   average_pfc_ = mean(V_spks[targ_ar[2][:],:],dims=1);
   average_str1_ = mean(V_spks[str_ar[1],:],dims=1)	
   average_str2_ = mean(V_spks[str_ar[2],:],dims=1)	
end

# ╔═╡ 74e400ea-dfc9-4e37-a954-51f6394ae611
mean(average_VC_[1:600])

# ╔═╡ fab60196-df31-4218-8e4e-f298b7ad730b
begin
    fs = 1000;
	#avec = [ii*10+1 for ii = 600:1600]
	avec = [ii*10+1 for ii = 1:600]
	
	periodogram_estimation = periodogram(average_VC_[1,avec], fs=fs)
	#periodogram_estimation = welch_pgram(average1[1,avec], fs=fs)
        pxx_VC = periodogram_estimation.power
        f_VC = periodogram_estimation.freq

	periodogram_estimation2 = periodogram(average_pfc_[1,avec], fs=fs)
#	periodogram_estimation2 = welch_pgram(average2[1,avec], fs=fs)
        pxx_pfc = periodogram_estimation2.power
        f_pfc = periodogram_estimation2.freq

	#periodogram_estimation3 = welch_pgram(average3[1,avec], fs=fs)
	periodogram_estimation3 = periodogram(average_str1_[1,avec], fs=fs)
        pxx_str1 = periodogram_estimation3.power
        f_str1 = periodogram_estimation3.freq
	#periodogram_estimation4 = welch_pgram(average4[1,avec], fs=fs)
	periodogram_estimation4 = periodogram(average_str2_[1,avec], fs=fs)
        pxx_str2 = periodogram_estimation4.power
        f_str2 = periodogram_estimation4.freq
	#periodogram_estimation5 = welch_pgram(average5[1,avec], fs=fs)
	periodogram_estimation5 = periodogram(average_thal1[1,avec], fs=fs)
        pxx_thal1 = periodogram_estimation5.power
        f_thal1 = periodogram_estimation5.freq

	periodogram_estimation6 = periodogram(average_thal2[1,avec], fs=fs)
        pxx_thal2 = periodogram_estimation6.power
        f_thal2 = periodogram_estimation6.freq
end

# ╔═╡ fa2538b2-c6e2-4ef3-9972-bace6f4ce689
begin
plot(f_pfc,pxx_pfc,xlims=[0,60],ylims=[0,5e-5]);
#plot(f_VC,pxx_VC,xlims=[0,60],ylims=[0,2e-6]);
#plot(f_str1[:],pxx_str1[:],color = "red",xlims=[0,60])#,ylims=[0,5])	
#plot!(f_str2[:],pxx_str2[:],color = "red",xlims=[0,60])#,ylims=[0,2],xlims=[0,60])	
#plot!(f_thal2[:],pxx_thal2[:],ylims=[0,1],xlims=[0,30])	
#plot!(f4[2:50],pxx4[2:50],color = "green", ylims=[0,3])		
end

# ╔═╡ ca342ef0-e40c-4819-bbb0-cad03672f6b0
begin
average_pfc__ = (round.(average_pfc_ .*100))./100
average_str__ = (round.(average_str2_ .*100))./100	
	
	sp_vec = [ii*10+1 for ii = 0:1600]
	avg_pfc_rnd = average_pfc__[:,sp_vec]
	avg_str_rnd = average_str__[:,sp_vec]
"""
	open("pfc_test6.txt", "w") do io
          writedlm(io, avg_pfc_rnd, ",")
	end
	open("str_test6.txt", "w") do io
          writedlm(io, avg_str_rnd, ",")
	end
		 """ 
end

# ╔═╡ 0d111ef5-7536-4c42-a704-4df72c5d41fd
plot(soll2.t,GG[targ_ar[2][:],:]',legend=false)

# ╔═╡ bdf18d67-d57a-45f2-9f49-c9731adee5d6
begin
plot(soll2.t,[VV[targ_ar[1][1:125],:]'],legend=false,yticks=[],color = "blue",size = (1000,700))
plot!(soll2.t,[VV[inhib_ar[1][1:25],:]'],legend=false,yticks=[],color = "red")
#plot!(soll2.t,[VV[inhib_mod_ar[1],:]],legend=false,yticks=[],color = "green")
	
end

# ╔═╡ 859f3376-c2d9-4808-9cac-9b5fda8d89ff
begin
plot(soll2.t,[VV[targ_ar[1][126:225],:]'],legend=false,yticks=[],color = "blue",size = (1000,700))
plot!(soll2.t,[VV[inhib_ar[1][26:45],:]'],legend=false,yticks=[],color = "red")
plot!(soll2.t,[VV[inhib_mod_ar[1],:]],legend=false,yticks=[],color = "green")
	
end

# ╔═╡ 3fd3d6ae-93d0-44a9-a0cb-b69177c6af3d
plot(soll2.t,V[inhib_ff[2],:],color = "red",xlims=(0,500))

# ╔═╡ 61ffbeeb-1506-47c9-a686-e7db0e4e083b
plot(soll2.t,V[TAN_ar[1],:],color = "red",xlims=(0,1000))

# ╔═╡ 3d7e48df-1efa-425f-871f-933e0c13e935
tt=soll2.t

# ╔═╡ 947acc13-0941-4922-aa9d-e6d66be9d2c7
 plot(soll2.t,[V[targ_ar[2][21],:]],xlims=(0,500))

# ╔═╡ 5824a794-bd0d-44d8-a9f2-c5761b496ac9
plot(soll2.t,[GGG[inhib_ar[2][5],:]])

# ╔═╡ 63cd9e70-a580-489d-9897-2fb127ef7c35
begin
pl1 = plot(soll2.t,[VV[targ_ar[2][1:100],:]'],legend=false,yticks=[],color = "blue",size = (1000,700),xlims=(0,1600));
#pl1 = plot(sol.t,[VV[48+1:48+48,:]'],legend=false,yticks=[])

plot!(pl1,soll2.t,[VV[inhib_ar[2][1:end],:]'],legend=false,yticks=[],color = "red",ylabel = "single neuron \n activity")
plot!(pl1,soll2.t,[VV[inhib_mod_ar[2],:]],legend=false,yticks=[],color = "green")	
end

# ╔═╡ 97f87264-8ae1-4944-bb69-10af7e0cc197
begin
plot(soll2.t,average_pfc_',xlims=(0,1600))#,ylims=(0,0.2));
#plot(soll2.t,average_VC_')	
#plot!(soll2.t,(average_str2_')./1)#,ylims=(-80,-39),color = "red")
#plot!(soll2.t,average_thal2',ylims=(-80,-39),color = "green",legend=false)
#plot!(soll2.t,average5',ylims=(-80,-39),legend=false)	
end

# ╔═╡ e7ce9324-f01e-44a5-b7ed-b4ddf867c1b1
begin
#plot(soll2.t,average_pfc_',xlims=(0,1600))#,ylims=(0,0.2));
#plot(soll2.t,average_VC_')	
plot(soll2.t,(average_str2_')./1)#,ylims=(-80,-39),color = "red")
#plot!(soll2.t,average_thal2',ylims=(-80,-39),color = "green",legend=false)
#plot!(soll2.t,average5',ylims=(-80,-39),legend=false)	
end

# ╔═╡ dcc285b4-8c8e-49ea-93ad-565c6340549c
begin
	plot(soll2.t, VV[str_ar[1],:]',legend=false,yticks=[],color = "blue",size = (1000,700) )
	plot!(soll2.t, VV[str_ar[2],:]',legend=false,yticks=[],color = "green",size = (1000,700) )
end

# ╔═╡ 70e25808-5110-4043-80ac-4311b0d7f553
begin
	plot(soll2.t, VV[GPi_ar[1],:]',legend=false,yticks=[],color = "blue",size = (1000,700) )
	plot!(soll2.t, VV[GPi_ar[2],:]',legend=false,yticks=[],color = "green",size = (1000,700) )
end

# ╔═╡ 1ae5df31-8f17-4eda-8c79-a1c9856efca9
begin
	plot(soll2.t[:], VV[thal_ar[1],:]',legend=false,yticks=[],color = "blue",size = (1000,700) )
	plot!(soll2.t[:], VV[thal_ar[2],:]',legend=false,yticks=[],color = "green",size = (1000,700))
end

# ╔═╡ af7cbb41-ebce-4376-be34-133f966e3a6b
begin
	plot(soll2.t[:], VV[STN_ar[1],:]',legend=false,yticks=[],color = "blue",size = (1000,700) )
	plot!(soll2.t[:], VV[STN_ar[2],:]',legend=false,yticks=[],color = "green",size = (1000,700))
end

# ╔═╡ 046b369e-3c97-4739-a78d-57c0eafde019
plot(soll2.t,average_GPi1')

# ╔═╡ fef0b413-dd86-41ee-9256-6f066ca9b318
rp

# ╔═╡ 5fe6446c-acff-482e-a7a0-3d81b563ae35
aa = ascending_input.(tt,16.0,rp,0.8)


# ╔═╡ 25ac56d2-e27c-498a-9361-06bdfd45dd6d
plot(soll2.t,aa,xlims=(0,500))

# ╔═╡ bd4ef1f4-61fa-4094-a0e2-5ddb24fc60fd
begin

# testing for beta offset



global pfc_LSS_t = zeros(100,1600*10+1)
global str_LSS_t = zeros(100,1600*10+1)		
global VC_LSS_t =  zeros(100,1600*10+1)		
	
	beta_t=	0

if beta_t==1	


	
for tloop=1:100	

	@info tloop
prob_param=copy(prob.p)	
if  rand()<=0.5 
	 ip = ptrn[:,rand(1:512)]
	 prob_param[str_ind2] .= -2	
	 
	else
	  ip = ptrn[:,512+rand(1:512)]
	  prob_param[str_ind1] .= -2	
end

I_in_t = zeros(Nrns);
I_in_t[targ_ar[1]] = 2*7*ip;





prob_param[cc] = I_in_t[targ_ar[1]]

prob_param[phase_ind] = rand()*2*pi  
        
prob_t = remake(prob;p=prob_param,u0=initial_state,tspan=(0, 1600));
solt_t = solve(prob_t,Vern7(),saveat = 0.1)
sol_t = convert(Array,solt_t);


 V_t=zeros(Nrns,length(solt_t.t));
	
	for ii = 1:Nrns
		
	   	V_t[ii,:] =  sol_t[(((ii-1)*7)+1),1:end];
		
	end

	V_t_spks=(sign.(V_t .+30) .+1) ./2

   


	#average_VC_t = mean(V_t_spks[targ_ar[1][:],:],dims=1);
   average_pfc_t = mean(V_t_spks[targ_ar[2][:],:],dims=1);
   average_str1_t = mean(V_t_spks[str_ar[1],:],dims=1)	
   average_str2_t = mean(V_t_spks[str_ar[2],:],dims=1)	
   average_str_t = (average_str1_t + average_str2_t) ./2	
   average_VC_t =  mean(V_t_spks[targ_ar[1][:],:],dims=1);

   pfc_LSS_t[tloop,:] = average_pfc_t;
   str_LSS_t[tloop,:] = average_str_t;	 
   VC_LSS_t[tloop,:] = average_VC_t; 	

end	

	

	
	
  	
	
end
end	


# ╔═╡ 9f53fe69-a9eb-41fd-917f-11a324bd9dee
begin
pfc_LSS_t_ = (round.(pfc_LSS_t .*100))./100 
	str_LSS_t_ = (round.(str_LSS_t .*100))./100 
	VC_LSS_t_ = (round.(VC_LSS_t .*100))./100 
	
	bvec_t = [ii*10+1 for ii = 0:1600]
	pfc_avg_rnd_t = pfc_LSS_t_[:,bvec_t]
	str_avg_rnd_t = str_LSS_t_[:,bvec_t]
    VC_avg_rnd_t = VC_LSS_t_[:,bvec_t]
	"""
open("pfc_test_thal_intra.txt", "w") do io
          writedlm(io, pfc_avg_rnd_t, ",")
	end

	open("VC_test_thal_intra.txt", "w") do io
          writedlm(io, VC_avg_rnd_t, ",")
	end

	open("str_test_thal_intra.txt", "w") do io
          writedlm(io, str_avg_rnd_t, ",")
	end
"""
end	

# ╔═╡ 00ce3408-fbbf-419f-a5d3-10ea63988802
begin
	global adj_rec=zeros(Nrns,Nrns,700)
	global str_rec1=zeros(length(targ_ar[2]),700)
	global str_rec2=zeros(length(targ_ar[2]),700)

	str_ar_tot = vcat(str_ar[1],str_ar[2])
	global pfc_spike_rec=zeros(length(targ_ar[2]),700)
    global pfc_spike_shrt_rec=zeros(length(targ_ar[2]),700)
	global str_spike_rec=zeros(length(str_ar_tot),700)
	global str_spike_shrt_rec=zeros(length(str_ar_tot),700)
	global image_sn=zeros(700)
	
	global spt_rec=Vector{Vector{Float32}}(undef,100*700)

	global g_thal1_pfc_rec = zeros(700,16001)
	global g_thal2_pfc_rec = zeros(700,16001)

	global cort_cort_wt_rec = zeros(700)
	global thal1_cort_wt_rec = zeros(700)
	global thal2_cort_wt_rec = zeros(700)
	
end

# ╔═╡ 8bc64567-0014-428e-94ea-4163bc710251
begin

cort_learning=0

	
	
spk_count1 = zeros(700,100)
spk_count2 = zeros(700,100)
catg_count = zeros(2)
I_in_arr3 = zeros(700,225)

if cort_learning ==1	
#goto
for loop3 =1:700

	@info loop3

	
	
if  rand()<=0.5 
	 sn=rand(1:512)
	 #input_pattern3 = ptrn[:,rand(1:512)]
	 input_pattern3 = ptrn[:,sn]
	 catg =1
	 catg_count[1] = catg_count[1] +1
	else
	 sn=512+rand(1:512)	
	 #input_pattern3 = ptrn[:,512+rand(1:512)]
	 input_pattern3 = ptrn[:,sn]	
	 catg =2
	 catg_count[2] = catg_count[2] +1
end
image_sn[loop3]=sn	
#####################################################################	
#stim on
I_in4 = zeros(Nrns);
I_in4[targ_ar[1]] = 2*7*input_pattern3;


prob_param=copy(prob.p)

adj_vec = vec(adj3)
syn_vec = vec(syn)	
conn_ind = findall(x-> x>0,syn_vec)

prob_param[cc] = I_in4[targ_ar[1]]
prob_param[dd] = adj_vec[conn_ind]
prob_param[phase_ind] = rand()*2*pi  
        
prob3 = remake(prob;p=prob_param,u0=initial_state,tspan=(0, 90));
solt3 = solve(prob3,Vern7(),saveat = 0.1)
sol3 = convert(Array,solt3);
break_state = sol3[:,end]


decision_ar = zeros(length(targ_ar))
spks = Vector{Vector{Float64}}(undef,length(targ_ar))
spks[1] = zeros(length(targ_ar[2]))
spks_thal = Vector{Vector{Float64}}(undef,length(thal_ar))
spks_short = Vector{Vector{Float64}}(undef,length(targ_ar))
spks_short[1] = zeros(length(targ_ar[2]))
spks_short2 = Vector{Vector{Float64}}(undef,length(targ_ar))
spks_short2[1] = zeros(length(targ_ar[2]))	
decision=0

		ii=2

			
		    spks_short[ii] = zeros(length(targ_ar[ii]))
		
		for ll = 1:length(targ_ar[ii])
		
		    v_short = sol3[(((targ_ar[ii][ll]-1)*7)+1),1:end]
            
			
		    if maximum(v_short)> 0 
		       tt = findpeaks(v_short,solt3.t,min_height=0.0)
		       spks_short[ii][ll] = length(tt)
			   pfc_spike_shrt_rec[ll,loop3] = length(tt)	

				if catg ==1
			       indx = convert(Int64,catg_count[1])
			       spk_count1[indx,ll] = length(tt)
		         else
			        indx = convert(Int64,catg_count[2])
			        spk_count2[indx,ll] = length(tt)
		         end

			   
				
		
			end
		
		
		end

	   #spiking of str
	str_ar_tot = vcat(str_ar[1],str_ar[2])
       for ll = 1:length(str_ar_tot)
	         v_short = sol3[(((str_ar_tot[ll]-1)*7)+1),1:end]
	         if maximum(v_short)> 0 
		       tt = findpeaks(v_short,solt3.t,min_height=0.0)
		       str_spike_shrt_rec[ll,loop3] = length(tt)

			 end
	   
	   
	   end
	   

#determination of tan suppression and DA uptick

    global tan_amp=100
	
	
	global	tan_input1_short = minimum([tan_amp ./(sum(spks_short[ii] .* str_in[ii][1][1:end])+sum(spks_short[ii] .* str_in[ii][2][1:end])),tan_amp]) 
		
    global  tan_input2_short = minimum([tan_amp ./(sum(spks_short[ii] .* str_in[ii][2][1:end])+sum(spks_short[ii] .* str_in[ii][1][1:end])),tan_amp])

	cort_str_in[loop3] = sum(spks_short[ii] .* str_in[ii][1][1:end])+sum(spks_short[ii] .* str_in[ii][2][1:end]) 	
			
###############################################################################
 # TAN pause period   	
    
 prob_param[tan_ind] .= 1.5#*tan_input1_short/tan_amp

prob4 = remake(prob;p=prob_param,u0=break_state,tspan=(90.1, 180));	
solt4 = solve(prob4,Vern7(),saveat = 0.1)
sol4 = convert(Array,solt4);
break_state2 = 	sol4[:,end]
sol3 = hcat(sol3,sol4)
time = vcat(solt3.t,solt4.t)			


			spks_short2[ii] = zeros(length(targ_ar[ii]))
		
		for ll = 1:length(targ_ar[ii])
		
		    v_short = sol4[(((targ_ar[ii][ll]-1)*7)+1),1:end]
            
			
		    if maximum(v_short)> 0 
		       tt = findpeaks(v_short,solt4.t,min_height=0.0)
		       spks_short2[ii][ll] = length(tt)

			end
		
		
		end
			
		
	des1 = sum(spks_short2[ii] .* str_in[ii][1][1:end]) +  rand(Poisson(tan_input1_short))
	des2 = sum(spks_short2[ii] .* str_in[ii][2][1:end]) +  rand(Poisson(tan_input2_short))

			if loop3<=30
              corr_dop = length(findall(x-> x>0,trial[1:loop3]))
            else
              corr_dop = length(findall(x-> x>0,trial[(loop3-30):loop3-1]))
            end

	println([sum(spks_short2[ii] .* str_in[ii][1][1:end]),sum(spks_short2[ii] .* str_in[ii][2][1:end]), des1, des2])
"""
	tan_input_ar1[loop3] = tan_input1_short
	tan_input_ar2[loop3] = tan_input2_short
	spike_input_ar1[loop3] = sum(spks_short[ii] .* str_in[ii][1][1:end])
    spike_input_ar2[loop3] = sum(spks_short[ii] .* str_in[ii][2][1:end])
"""   
    des_ar1[loop3]=des1
    des_ar2[loop3]=des2
	dop_ar[loop3]=corr_dop		


	if des1>=des2   #decision from each block
			decision_ar[ii] =1
		    decision=1
		    trial_dec[loop3]=1
	else
			decision_ar[ii] =2
		    decision=2
		    trial_dec[loop3]=2
	end
		@info decision_ar[ii]	
  


#######################################################################	
#TAN pause ends
	
# one striatal block activated and performs lateral inhibition on other striatal blocks
if decision==1	
prob_param[str_ind2] .= -2	#shut off str 2
else
prob_param[str_ind1] .= -2	#shut off str 1	
end	
prob_param[tan_ind] .= 1.5
	
prob5 = remake(prob;p=prob_param,u0=break_state2,tspan=(180.1, 600));	
solt5 = solve(prob5,Vern7(),saveat = 0.1)
sol5 = convert(Array,solt5);
break_state3 = 	sol5[:,end]
sol3 = hcat(sol3,sol5)
time = vcat(time,solt5.t)	
	
#########################################################################
#stim off

I_in4[targ_ar[1]] = 0*input_pattern3;	
prob_param[cc] = I_in4[targ_ar[1]]
prob6 = remake(prob;p=prob_param,u0=break_state3,tspan=(600.1, 1600));	
solt6 = solve(prob6,Vern7(),saveat = 0.1)
sol6 = convert(Array,solt6);
sol3 = hcat(sol3,sol6)
time = vcat(time,solt6.t)	

str_ar_tot = vcat(str_ar[1],str_ar[2])
V_pfc=zeros(length(targ_ar[2]),length(time))
V_str=zeros(length(str_ar_tot),length(time))
g_thal1_pfc=zeros(length(targ_ar[2]),length(time))
g_thal2_pfc=zeros(length(targ_ar[2]),length(time))	
	
	for ii = 2:length(targ_ar)

		#decision making
		#catg1=zeros(length(targ_ar[ii])) # weighted input
		#catg1=zeros(length(targ_ar[ii]))
		spks[ii] = zeros(length(targ_ar[ii]))
		
	#saving pfc activities	and g_thal_pfc
		for ll = 1:length(targ_ar[ii]) 
		
		    v = sol3[(((targ_ar[ii][ll]-1)*7)+1),1:end]

			thal_in1 = sum(adj3[targ_ar[ii][ll],thal_ar[1]])
			thal_in2 = sum(adj3[targ_ar[ii][ll],thal_ar[2]])
			
			
			gstp = sol3[(((targ_ar[ii][ll]-1)*7)+7),1:end]
		#	v = sol3[(((ll-1)*7)+1),1:end]
            V_pfc[ll,:] = v
            g_thal1_pfc[ll,:] = gstp*thal_in1
            g_thal2_pfc[ll,:] = gstp*thal_in2 
			
		    if maximum(v)> 0 
		       tt = findpeaks(v,time,min_height=0.0)
		       spks[ii][ll] = length(tt)
               pfc_spike_rec[ll,loop3] = length(tt) 
			   spt_rec[(loop3-1)*100+ll] = tt
				
			else
			   spt_rec[(loop3-1)*100+ll] = zeros(10)
				
		
			end
		
		
		end
#saving str activities
		for nn = 1:length(str_ar_tot) 
			v = sol3[(((str_ar_tot[nn]-1)*7)+1),1:end]
            V_str[nn,:] = v
			if maximum(v)> 0 
		       tt = findpeaks(v,time,min_height=0.0)
		       str_spike_rec[nn,loop3] = length(tt)

			end
		
		end

		

	pfc_avg[loop3,:] = mean(V_pfc,dims=1);
	str_avg[loop3,:] = mean(V_str,dims=1);

	V_pfc_spks=(sign.(V_pfc .+30) .+1) ./2	
	V_str_spks=(sign.(V_str .+30) .+1) ./2

	pfc_LSS[loop3,:] = mean(V_pfc_spks,dims=1)	
	str_LSS[loop3,:] = mean(V_str_spks,dims=1)		

	tan_amp=100

	#global	tan_input1 = minimum([tan_amp ./(sum(spks[ii] .* str_in[ii][1][1:end])+sum(spks[ii] .* str_in[ii][2][1:end])),tan_amp]) 
		
   # global   tan_input2 = minimum([tan_amp ./(sum(spks[ii] .* str_in[ii][2][1:end])+sum(spks[ii] .* str_in[ii][1][1:end])),tan_amp]) 

   tan_input_ar1[loop3] = tan_input1_short
	tan_input_ar2[loop3] = tan_input2_short
	spike_input_ar1[loop3] = sum(spks_short2[ii] .* str_in[ii][1][1:end])
    spike_input_ar2[loop3] = sum(spks_short2[ii] .* str_in[ii][2][1:end])
	
	
	
	
	end	

    
	
 #feedback driven learing
  for ii = 2:(length(targ_ar))	

		 #cortical learning
	  for kk = 1:length(targ_ar[ii])
		 
	   if decision == catg 
		if spks[ii][kk]>0  

			
	      
			preblock = targ_ar[ii-1]
				
		
			
			source = adj3[targ_ar[ii][kk],preblock]
			source_con = findall(x-> x>0, source)
			pres = preblock[source_con]
			
			for jj = 1:length(pres)
				pre_v = sol3[(((pres[jj]-1)*6)+1),1:end]
			  
			  if (maximum(pre_v)>0) 	  
				pre_tt = findpeaks(pre_v,time,min_height=0.0)
			  
				   adj3[targ_ar[ii][kk],pres[jj]] = adj3[targ_ar[ii][kk],pres[jj]] + ((1/200)*spks[ii][kk]*length(pre_tt)/100)*maximum([(5-adj3[targ_ar[ii][kk],pres[jj]]),0])
				  
            #       adj3[targ_ar[ii][kk],pres[jj]] = adj3[targ_ar[ii][kk],pres[jj]] + 0.1*((1/100)*spks[ii][kk]*length(pre_tt)/100)*(5-adj3[targ_ar[ii][kk],pres[jj]])
				  
				  
			  end
			end

            thal_ar_tot = vcat(thal_ar[1],thal_ar[2])
			thal_source = adj3[targ_ar[ii][kk],thal_ar_tot]
			thal_source_con = findall(x-> x>0, thal_source)
			thal_pres = thal_ar_tot[thal_source_con]
				  
			#if decision==1
			# thal_source = adj3[targ_ar[ii][kk],thal_ar[1]]
			# thal_source_con = findall(x-> x>0, thal_source)
			# thal_pres = thal_ar[1][thal_source_con]	
			#else
			# thal_source = adj3[targ_ar[ii][kk],thal_ar[2]] 
			# thal_source_con = findall(x-> x>0, thal_source)
			# thal_pres = thal_ar[2][thal_source_con]	
			#end
				
			
			

			for jj = 1:length(thal_pres)
				pre_v = sol3[(((thal_pres[jj]-1)*7)+1),1:end]
			 # if (adj3[targ_ar[ii][kk],thal_pres[jj]]<10) && (maximum(pre_v)>0) 
			  if (maximum(pre_v)>0) 	  
				pre_tt = findpeaks(pre_v,time,min_height=0.0)
			  
				#adj3[targ_ar[ii][kk],thal_pres[jj]] = adj3[targ_ar[ii][kk],thal_pres[jj]] + 0.5*0.25*0.1*((1/100)*spks[ii][kk]*length(pre_tt)/500)*maximum([(5-adj3[targ_ar[ii][kk],thal_pres[jj]]),0])

				adj3[targ_ar[ii][kk],thal_pres[jj]] = adj3[targ_ar[ii][kk],thal_pres[jj]] + 1/1*((1/200)*spks[ii][kk]*length(pre_tt)/500)*maximum([(6-adj3[targ_ar[ii][kk],thal_pres[jj]]),0])  

				#  if adj3[targ_ar[ii][kk],thal_pres[jj]] >1.5

				#	  adj3[targ_ar[ii][kk],thal_pres[jj]]=1.5
				  
				#  end
				  
		      end
			end

			
         #end

	 
	    end

	   end	
   end

# feedback driven striatal leaning		
#pred_amp = maximum([cort_str_in[1],1])
#pred =  1 - minimum([pred_amp/cort_str_in[loop3],1]) 
pred =  1 - minimum([3/cort_str_in[loop3],1]) 	  
T_ltd = -0.25 # between -1 to 0	  


   for jj = 1:length(targ_ar[ii])
	   str_in[ii][1][jj] = str_in[ii][1][jj]*0.99
	   str_in[ii][2][jj] = str_in[ii][2][jj]*0.99

	   if spks[ii][jj]>0

          if decision == catg #dopamine release
             """
			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = minimum([str_in[ii][1][jj] + str_lr*(0.1*minimum([spks[ii][jj],200])/20)*(tan_input1)/100,1])
			  else
				  str_in[ii][2][jj] = minimum([str_in[ii][2][jj] + str_lr*(0.1*minimum([spks[ii][jj],200])/20)*(tan_input2)/100,1])
			
			  end
               """

			  D_in = 1
			  pred_err = D_in - pred
			#  str_lr = CS_plast(pred_err,T_ltd)
			  str_lr = CS_plast3(pred_err)
			  
			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = minimum([str_in[ii][1][jj] + str_lr*0.001*minimum([spks[ii][jj],200]),1])
			  else
				  str_in[ii][2][jj] = minimum([str_in[ii][2][jj] + str_lr*0.001*minimum([spks[ii][jj],200]),1])
			
			  end
			  
	   
		  else #dopamine not released
             """
			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = maximum([str_in[ii][1][jj] - str_lr*(0.4*spks[ii][jj]/20)*(tan_input1)/100,0])
			
			  else
				  str_in[ii][2][jj] = maximum([str_in[ii][2][jj] - str_lr*(0.4*spks[ii][jj]/20)*(tan_input2)/100,0])
			
			  end
             """
			  #expec = maximum([100-tan_input1_short,0])
			  #ltd = (expec*(expec-100))/50
			  #ltd = expec
              D_in = 0
			  pred_err = D_in - pred
			 # str_lr = CS_plast(pred_err,T_ltd)
			  str_lr = CS_plast3(pred_err)
			  
			  
			  if decision_ar[ii] == 1
				  str_in[ii][1][jj] = maximum([str_in[ii][1][jj] + str_lr*0.001*minimum([spks[ii][jj],200]),0])#0.002
			
			  else
				  str_in[ii][2][jj] = maximum([str_in[ii][2][jj] + str_lr*0.001*minimum([spks[ii][jj],200]),0])#0.002
			
			  end


		  end
#no spiking
		#else
        #      str_in[ii][1][jj] = str_in[ii][1][jj]*0.95
		#	  str_in[ii][2][jj] = str_in[ii][2][jj]*0.95
			
	   
	   end

	   

	adj3[str_ar[1],targ_ar[ii][jj]] = sign.(adj3[str_ar[1],targ_ar[ii][jj]])*1*(str_in[ii][1][jj]+0.0)
	   
    adj3[str_ar[2],targ_ar[ii][jj]] = sign.(adj3[str_ar[2],targ_ar[ii][jj]])*1*(str_in[ii][2][jj]+0.0)

	   
      
   end

	if (decision == catg) 
		@info "Correct!"
		trial[loop3] = 1

	else
        @info "Incorrect!"

	end	  

  	  
	
	  
end
global adj_rec[:,:,loop3] = copy(adj3);
global str_rec1[:,loop3] = copy(str_in[2][1]);
global str_rec2[:,loop3] = copy(str_in[2][2])	

global g_thal1_pfc_rec[loop3,:] = sum(g_thal1_pfc,dims=1)	
global g_thal2_pfc_rec[loop3,:] = sum(g_thal2_pfc,dims=1)	
global cort_cort_wt_rec[loop3] = sum(adj3[targ_ar[2],targ_ar[1]])/(225*8)
global thal1_cort_wt_rec[loop3] = sum(adj3[targ_ar[2],thal_ar[1]])/800	
global thal2_cort_wt_rec[loop3] = sum(adj3[targ_ar[2],thal_ar[2]])/800		
#global sol_save = copy(sol3)
#global time_save = copy(time)	
	
# tan_input[1] = tan_input[1]*0.95
plot(spike_input_ar1[1:end],labels="cortico-striatal input 1")
plot!(spike_input_ar2[1:end],labels="cortico-striatal input 2")
plot!(tan_input_ar1[1:end],labels="TAN activity ",xlabel="trials",ylims=(0,100))
	
end
end
end

# ╔═╡ 4690e30e-8c91-444d-bc46-7b476b0f1ca5
begin
plot(soll2.t,g_thal1_pfc_rec[660:700,:]',legend=false)
	plot!(soll2.t,g_thal1_pfc_rec[1:150,:]')
	
end

# ╔═╡ eb3e4c23-c237-481d-ad99-2ef865675a3d
begin
	plot(cort_cort_wt_rec)
	plot!(thal1_cort_wt_rec)
plot!(thal2_cort_wt_rec)
end

# ╔═╡ 24cb8960-69d4-4fa2-96cc-19f69367da2a
begin
	
	plot(spike_input_ar1[1:end],labels="cortico-striatal input 1")
	plot!(spike_input_ar2[1:end],labels="cortico-striatal input 2")
	plot!(tan_input_ar1[1:end],labels="TAN activity ",xlabel="trials",ylims=(0,100))#,xlims=(0,10))
end

# ╔═╡ ca497ba0-3f9d-4c15-96cb-94e40eb48fda
plot(dop_ar[30:end]./30,legend=false,xlabel="trials",ylabel="accuracy",ylims=(0,1))

# ╔═╡ 04a89c6a-b549-4d29-ba07-c05b3ca8c8eb
plot(soll2.t,pfc_LSS'[:,645],legend=false)

# ╔═╡ 4d221955-5662-4136-9d70-7649717e58bc
plot(soll2.t,str_LSS'[:,675],legend=false)

# ╔═╡ 3e979bd4-a173-44fd-9a5e-c9b04a718827
Gray.(trial_dec[1:700]./2)

# ╔═╡ 4cae57cb-56cb-4158-8982-c8f6b90af828
Gray.(trial[1:700])

# ╔═╡ 7e2c891d-b135-4613-81ec-5a7a73f33f82
begin
	pfc_avg_ = (round.(pfc_avg .*100))./100
	str_avg_ = (round.(str_avg .*100))./100

	pfc_LSS_ = (round.(pfc_LSS .*100))./100 
	str_LSS_ = (round.(str_LSS .*100))./100 
	
	bvec = [ii*10+1 for ii = 0:1600]
	pfc_avg_rnd = pfc_avg_[:,bvec]
	str_avg_rnd = str_avg_[:,bvec]

	pfc_LSS_rnd = pfc_LSS_[:,bvec]
	str_LSS_rnd = str_LSS_[:,bvec]

	g_thal1_pfc_rec_ = (round.(g_thal1_pfc_rec .*100))./100
	g_thal2_pfc_rec_ = (round.(g_thal2_pfc_rec .*100))./100
	
	g_thal1_rnd = g_thal1_pfc_rec_[:,bvec]
	g_thal2_rnd = g_thal2_pfc_rec_[:,bvec]

	adj_rec_rnd = (round.(adj_rec .*100))./100
	lc= length(targ_ar[2])
    adj_rec_cmprsd = zeros(lc*100,Nrns)
	for ii = 1:100
         tp = ii*7
		 adj_rec_cmprsd[((ii-1)*lc+1):ii*lc,:] .= adj_rec[targ_ar[2],:,tp]
	
	end
end

# ╔═╡ 414e7b14-458b-4cf8-bd0f-8b0d4c8cc993
begin
"""

	open("perf64.txt", "w") do io
          writedlm(io, trial, ",")
	end

	open("dec64.txt", "w") do io
          writedlm(io, trial_dec, ",")
	end

	open("pfc64.txt", "w") do io
          writedlm(io, pfc_avg_rnd, ",")
	end
	
   open("str64.txt", "w") do io
          writedlm(io, str_avg_rnd, ",")
	end

   	open("pfcLSS64.txt", "w") do io
          writedlm(io, pfc_LSS_rnd, ",")
	end

	open("strLSS64.txt", "w") do io
          writedlm(io, str_LSS_rnd, ",")
	end

   open("full_circuit_adj64.txt", "w") do io
          writedlm(io, adj3, ",")
	   
	end

	open("tan_input64.txt", "w") do io
          writedlm(io, tan_input_ar1, ",")	
end

	open("CS1_input64.txt", "w") do io
          writedlm(io, spike_input_ar1, ",")	
end

	open("CS2_input64.txt", "w") do io
          writedlm(io, spike_input_ar2, ",")	
end

 open("pfc_spike_rec64.txt", "w") do io
          writedlm(io, pfc_spike_rec, ",")	
end

open("pfc_spike_shrt_rec64.txt", "w") do io
          writedlm(io, pfc_spike_shrt_rec, ",")	
end

	open("str_spike_rec64.txt", "w") do io
          writedlm(io, str_spike_rec, ",")	
end

	open("str_spike_shrt_rec64.txt", "w") do io
          writedlm(io, str_spike_shrt_rec, ",")	
end

open("spt_pattern64.txt", "w") do io
          writedlm(io, spt_rec, ",")#
	end

open("str_rec1_64.txt","w") do io
          writedlm(io,str_rec1,",")
end

open("str_rec2_64.txt","w") do io
          writedlm(io,str_rec2,",")
end

open("gthal1_64.txt","w") do io
          writedlm(io,g_thal1_rnd,",")
end

open("gthal2_64.txt","w") do io
          writedlm(io,g_thal2_rnd,",")
end

open("cort_cort_wt_64.txt","w") do io
          writedlm(io,cort_cort_wt_rec,",")
end

open("thal1_cort_wt_64.txt","w") do io
          writedlm(io,thal1_cort_wt_rec,",")
end

open("thal2_cort_wt_64.txt","w") do io
          writedlm(io,thal2_cort_wt_rec,",")
end

open("image_sn64.txt", "w") do io
          writedlm(io, image_sn, ",")	
end

"""
	#pfc_avg2 = readdlm("full_circuit_pfc_avg.txt",',')
    
	#str_avg2 = readdlm("full_circuit_str_avg.txt",',')

	#adj_mov2 = readdlm("adj_movie.txt",',')

end

# ╔═╡ 65c5dc63-241b-4525-816c-3f75899e4e34
begin
ppl1=plot(Gray.(adj3[targ_ar[2],thal_ar[1]]/6));
ppl2=plot(Gray.(adj3[targ_ar[2],thal_ar[2]]/6));
plot(ppl1,ppl2,layout=(1,2))	
end

# ╔═╡ 4b62d873-589c-4acc-87fc-b88275580a08
plot(Gray.(adj3[targ_ar[2],targ_ar[1]]/4))

# ╔═╡ c654cae2-c51b-401d-849d-9d1a32e9e214
maximum(adj3[targ_ar[2],targ_ar[1]])

# ╔═╡ 611c69f9-dc20-42de-a908-baf8edc7b6e4
maximum(adj3[targ_ar[2],thal_ar[1]])

# ╔═╡ 237b9b1d-b3a8-404d-bc12-fb91e49560d4
maximum(str_in[2][1])

# ╔═╡ f47560ab-24aa-4e3a-af31-7f8f83f869df
maximum(str_in[2][2])

# ╔═╡ b915d877-3298-4c27-9c5e-64129d6dc9b6
mean(str_in[2][1])

# ╔═╡ 67674543-4f7f-4557-900c-7183d8ce2f78
mean(str_in[2][2])

# ╔═╡ e6645293-3f5e-4094-9c4c-732de6c3ae08
Gray.(str_in[2][1]./maximum(str_in[2][1]))

# ╔═╡ 2aa3f3bd-c12c-4cf0-9d3a-058ca3095eb2
Gray.(str_in[2][2]./maximum(str_in[2][2]))

# ╔═╡ 7e6b443a-941b-43fd-a11b-4ecf10bcf78e
plot(Gray.(str_rec1[:,1:700]./maximum(str_rec1)))

# ╔═╡ 792289ee-d971-4362-9a6c-aac49e14d911
plot(Gray.(str_rec2[:,1:700]./maximum(str_rec2)))

# ╔═╡ Cell order:
# ╠═8872e8db-cb41-4068-8401-1721f07642c8
# ╠═406f6214-cb40-11ec-037a-1325bda2f580
# ╠═1a01f8a2-d779-4b64-9401-3e746acdd6ab
# ╠═abf532aa-d333-42fe-96de-b9ada89852e9
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
# ╠═64faf636-43c9-4aa9-8d78-e7aa4d91db50
# ╠═0a803feb-3dd1-43ac-9afc-1b0afd19ce2d
# ╠═738fb9f1-81f3-4738-a8bd-407461c9586f
# ╠═ca25e5b5-9c81-461f-b014-54221ffd06c6
# ╠═e4b89f6a-21f0-42ca-950c-448afa5ec535
# ╠═61c5b42a-8723-4334-a3ba-8c8558b11284
# ╠═233e2ddc-6148-459a-87ed-16646fea5316
# ╠═3be21966-09e5-46be-995c-c53e49d0a3c2
# ╠═c35348de-e55e-4c20-895e-f49a1bbeec4b
# ╠═95653cf9-4db8-4cf9-b38b-fcb639c03bd7
# ╠═ae38608c-2193-4439-b439-29fa7805c05f
# ╠═a42dcd5b-dc7b-47bf-8536-be6c21c2967b
# ╠═b37c39ea-6746-48a9-b450-b3ea25530e7f
# ╠═f2537041-c4d9-4f2f-be62-5c00a84f173d
# ╠═f5294dac-d33d-4d61-b901-af8ac2b61dfe
# ╟─b47ed6fb-82dc-4a1c-98bf-870089d2c9e9
# ╠═f2f4b6b3-9098-4dcb-ac10-b838af07980a
# ╠═88ebe172-46a3-4032-acf3-950e5d9ab7a6
# ╠═dc575aaf-887e-40e0-9e19-235e16532735
# ╠═6d7ce7e5-65d3-4cf1-ab27-221cb07dd4a8
# ╠═f18e3d9a-810f-4849-bbaa-4b6142dde625
# ╠═c0943891-c172-432b-bb2f-59dedcebc07d
# ╠═15b613ff-5edb-49a7-b770-a2afcd361091
# ╠═f7f439ef-ba85-4023-b478-3f095fd9ff5b
# ╠═092502cf-4a04-4d70-b954-f3dfd2a6c9fa
# ╠═e1932634-20f9-4281-bbf9-6e910fc5dd8b
# ╠═ec6d5982-6e80-48ed-9d49-edc7fb07888c
# ╠═9a3721f6-2b99-400e-af9d-c3969b57369a
# ╠═58478f96-2575-473c-bf54-4f34dcb421cc
# ╠═9f7fe203-ac41-4502-9b41-7028bf9f21b3
# ╠═225e1431-bf2a-4855-abd6-f0d826b5abe8
# ╠═86b87ae5-01c9-4fc8-b37a-e5bf8b8d9d72
# ╠═143ab0df-0fd1-4add-ae31-1b734eb86ef4
# ╠═0d111ef5-7536-4c42-a704-4df72c5d41fd
# ╠═bdf18d67-d57a-45f2-9f49-c9731adee5d6
# ╠═859f3376-c2d9-4808-9cac-9b5fda8d89ff
# ╠═3fd3d6ae-93d0-44a9-a0cb-b69177c6af3d
# ╠═61ffbeeb-1506-47c9-a686-e7db0e4e083b
# ╠═fef0b413-dd86-41ee-9256-6f066ca9b318
# ╠═3d7e48df-1efa-425f-871f-933e0c13e935
# ╠═5fe6446c-acff-482e-a7a0-3d81b563ae35
# ╠═25ac56d2-e27c-498a-9361-06bdfd45dd6d
# ╠═947acc13-0941-4922-aa9d-e6d66be9d2c7
# ╠═ed2e05a2-15eb-48d0-9cf7-5a0468a3f05d
# ╠═5824a794-bd0d-44d8-a9f2-c5761b496ac9
# ╠═63cd9e70-a580-489d-9897-2fb127ef7c35
# ╠═192e8ee4-aa06-412e-97ad-0197da60bc86
# ╠═97f87264-8ae1-4944-bb69-10af7e0cc197
# ╠═e7ce9324-f01e-44a5-b7ed-b4ddf867c1b1
# ╠═dcc285b4-8c8e-49ea-93ad-565c6340549c
# ╠═70e25808-5110-4043-80ac-4311b0d7f553
# ╠═1ae5df31-8f17-4eda-8c79-a1c9856efca9
# ╠═af7cbb41-ebce-4376-be34-133f966e3a6b
# ╠═046b369e-3c97-4739-a78d-57c0eafde019
# ╠═74e400ea-dfc9-4e37-a954-51f6394ae611
# ╠═fab60196-df31-4218-8e4e-f298b7ad730b
# ╠═ca342ef0-e40c-4819-bbb0-cad03672f6b0
# ╠═fa2538b2-c6e2-4ef3-9972-bace6f4ce689
# ╠═60cae459-fddc-435f-a07a-081c9aa54529
# ╠═bd4ef1f4-61fa-4094-a0e2-5ddb24fc60fd
# ╠═9f53fe69-a9eb-41fd-917f-11a324bd9dee
# ╠═a110ce12-bde6-401d-8aff-2dfbd99e57b5
# ╠═00ce3408-fbbf-419f-a5d3-10ea63988802
# ╠═8bc64567-0014-428e-94ea-4163bc710251
# ╠═4690e30e-8c91-444d-bc46-7b476b0f1ca5
# ╠═eb3e4c23-c237-481d-ad99-2ef865675a3d
# ╠═24cb8960-69d4-4fa2-96cc-19f69367da2a
# ╠═ca497ba0-3f9d-4c15-96cb-94e40eb48fda
# ╠═04a89c6a-b549-4d29-ba07-c05b3ca8c8eb
# ╠═4d221955-5662-4136-9d70-7649717e58bc
# ╠═3e979bd4-a173-44fd-9a5e-c9b04a718827
# ╠═4cae57cb-56cb-4158-8982-c8f6b90af828
# ╠═7e2c891d-b135-4613-81ec-5a7a73f33f82
# ╠═414e7b14-458b-4cf8-bd0f-8b0d4c8cc993
# ╠═65c5dc63-241b-4525-816c-3f75899e4e34
# ╠═4b62d873-589c-4acc-87fc-b88275580a08
# ╠═c654cae2-c51b-401d-849d-9d1a32e9e214
# ╠═611c69f9-dc20-42de-a908-baf8edc7b6e4
# ╠═237b9b1d-b3a8-404d-bc12-fb91e49560d4
# ╠═f47560ab-24aa-4e3a-af31-7f8f83f869df
# ╠═b915d877-3298-4c27-9c5e-64129d6dc9b6
# ╠═67674543-4f7f-4557-900c-7183d8ce2f78
# ╠═e6645293-3f5e-4094-9c4c-732de6c3ae08
# ╠═2aa3f3bd-c12c-4cf0-9d3a-058ca3095eb2
# ╠═7e6b443a-941b-43fd-a11b-4ecf10bcf78e
# ╠═792289ee-d971-4362-9a6c-aac49e14d911
