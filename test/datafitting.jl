using Neuroblox, Test, Graphs, MetaGraphs, OrderedCollections, LinearAlgebra, DataFrames
using MAT

### Load data ###
vars = matread(joinpath(@__DIR__, "spectralDCM_toydata.mat"));
data = DataFrame(vars["data"], :auto)   # turn data into DataFrame
x = vars["x"]                           # initial conditions
nrr = ncol(data)                         # number of recorded regions
max_iter = 126
########## assemble the model ##########

g = MetaDiGraph()
regions = Dict()
@parameters κ=0.0 [tunable = true]     # define brain-wide decay parameter for hemodynamics
for ii = 1:nrr
    region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
    add_blox!(g, region)
    regions[ii] = 2ii - 1    # store index of neural mass model
    # add hemodynamic observer
    observer = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=κ)
    add_blox!(g, observer)
    # connect observer with neuronal signal
    add_edge!(g, 2ii - 1, 2ii, Dict(:weight => 1.0))
end

# add symbolic weights
@parameters A[1:length(vars["pE"]["A"])] = vec(vars["pE"]["A"]) [tunable = true]
for (i, idx) in enumerate(CartesianIndices(vars["pE"]["A"]))
    if idx[1] == idx[2]
        add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, -exp(A[i])/2)  # treatement of diagonal elements in SPM12
    else
        add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
    end
end

# compose model
@named neuronmodel = system_from_graph(g)
neuronmodel = structural_simplify(neuronmodel)

# measurement model
@named bold = boldsignal()

# attribute initial conditions to states
all_s, idx_drive = get_states_without_drive(neuronmodel)
initcond = OrderedDict{typeof(all_s[1]), eltype(x)}()
rnames = []
map(x->push!(rnames, split(string(x), "₊")[1]), all_s); 
rnames = unique(rnames);
for (i, r) in enumerate(rnames)
    for (j, s) in enumerate(all_s[r .== map(x -> x[1], split.(string.(all_s), "₊"))])   # TODO: fix this solution, it is not robust!!
        initcond[s] = x[i, j]
    end
end

modelparam = OrderedDict()
for par in tunable_parameters(neuronmodel)
    modelparam[par] = Symbolics.getdefaultval(par)
end
np = length(modelparam)
params_idx = Dict(:evolpars => collect(1:np))
# Noise parameter mean
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
n = length(modelparam[:lnα]);
params_idx[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
params_idx[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = zeros(Float64, nrr);   # region specific observation noise
params_idx[:lnγ] = collect(np+1:np+nrr);
np += nrr
params_idx[:u_states] = idx_drive

for par in parameters(bold)
    modelparam[par] = Symbolics.getdefaultval(par)
end
# number params_idx of observation model parameters
nop = length(parameters(bold))
params_idx[:obspars] = np+1:np+nop

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:lnγ] = ones(Float64, nrr)./64.0;
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0; 
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
for (k, v) in paramvariance
    if occursin("A[", string(k))
        paramvariance[k] = vars["pC"][1, 1]
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperpriors = Dict(:Πλ_pr => vars["ihC"]*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => [vars["hE"]]              # prior metaparameter mean, needs to be a vector
                  );

csdsetup = Dict(:p => 8, :freq => vec(vars["Hz"]), :dt => vars["dt"]);

(state, setup) = setup_sDCM(data, neuronmodel, bold, initcond, csdsetup, priors, hyperpriors, params_idx);
for iter in 1:128
    state.iter = iter
    run_sDCM_iteration!(state, setup)
    print("iteration: ", iter, " - F:", state.F[end] - state.F[2], " - dF predicted:", state.dF[end], "\n")
    if iter >= 4
        criterion = state.dF[end-3:end] .< setup.tolerance
        if all(criterion)
            print("convergence\n")
            break
        end
    end
end
print("maxixmum iterations reached\n")

### COMPARE RESULTS WITH MATLAB RESULTS ###
@show state.F[end], vars["F"]
@test state.F[end] < vars["F"]*0.99
@test state.F[end] > vars["F"]*1.01