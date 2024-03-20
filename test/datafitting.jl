using Neuroblox, Test, Graphs, MetaGraphs, OrderedCollections, LinearAlgebra, DataFrames
using MAT

### Load data ###
vars = matread(joinpath(@__DIR__, "spectralDCM_toydata.mat"));
data = DataFrame(vars["data"], :auto)   # turn data into DataFrame
x = vars["x"]                           # initial conditions
nd = ncol(data)                         # number of dimensions

########## assemble the model ##########

g = MetaDiGraph()
regions = Dict()
@parameters κ=0.0 [tunable = true]     # define brain-wide decay parameter for hemodynamics
for ii = 1:nd
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
# We need to use split=false here because we rely on a specific parameter ordering
neuronmodel = structural_simplify(neuronmodel; split=false)

# measurement model
@named bold = boldsignal()

# attribute initial conditions to unknowns
all_s = unknowns(neuronmodel)
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
for par in parameters(neuronmodel)
    if istunable(par)
        modelparam[par] = Symbolics.getdefaultval(par)
    end
end

# Noise parameter mean
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
modelparam[:lnγ] = zeros(Float64, nd);   # region specific observation noise
modelparam[:C] = ones(Float64, nd);     # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM12...

for par in parameters(bold)
    modelparam[par] = Symbolics.getdefaultval(par)
end

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:C] = zeros(Float64, nd);
paramvariance[:lnγ] = ones(Float64, nd)./64.0;
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

params = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperparams = Dict(:Πλ_pr => vars["ihC"]*ones(1,1),   # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => [vars["hE"]]             # prior metaparameter mean, needs to be a vector
                  )

csdsetup = Dict(:p => 8, :freq => vec(vars["Hz"]), :dt => vars["dt"])
results = spectralVI(data, neuronmodel, bold, initcond, csdsetup, params, hyperparams)

### COMPARE RESULTS WITH MATLAB RESULTS ###
@show results.F, vars["F"]
@test results.F < vars["F"]*0.99
@test results.F > vars["F"]*1.01
