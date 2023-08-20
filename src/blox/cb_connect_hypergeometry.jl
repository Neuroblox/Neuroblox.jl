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