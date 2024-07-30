using Neuroblox, Test, Graphs, MetaGraphs, OrderedCollections, LinearAlgebra, DataFrames
using MAT

### Load data ###
vars = matread(joinpath(@__DIR__, "spectralDCM_toydata.mat"));
data = DataFrame(vars["data"], :auto)    # turn data into DataFrame, name column names after building the model.
x = vars["x"]                            # initial conditions
nrr = ncol(data)                         # number of recorded regions
max_iter = 128

########## assemble the model ##########
g = MetaDiGraph()
regions = Dict()
@parameters lnκ=0.0 [tunable = true] lnϵ=0.0 [tunable=true] C=1/16 [tunable = false]
for ii = 1:nrr
    region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
    add_blox!(g, region)
    regions[ii] = nv(g)    # store index of neural mass model
    taskinput = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
    add_blox!(g, taskinput)
    add_edge!(g, nv(g), nv(g) - 1, Dict(:weight => C))
    # add hemodynamic observer
    observer = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=lnκ, lnϵ=lnϵ)
    add_blox!(g, observer)
    # connect observer with neuronal signal
    add_edge!(g, nv(g) - 2, nv(g), Dict(:weight => 1.0))
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
neuronmodel = structural_simplify(neuronmodel; split=false)

# attribute initial conditions to states
sts, idx_sts = get_dynamic_states(neuronmodel)
idx_u = get_idx_tagged_vars(neuronmodel, "ext_input")         # get index of external input state
idx_bold, obsvars = get_eqidx_tagged_vars(neuronmodel, "measurement")  # get index of equation of bold state
rename!(data, Symbol.(obsvars))

initcond = OrderedDict(sts .=> 0.0)
rnames = []
map(x->push!(rnames, split(string(x), "₊")[1]), sts);
rnames = unique(rnames);
for (i, r) in enumerate(rnames)
    for (j, s) in enumerate(sts[r .== map(x -> x[1], split.(string.(sts), "₊"))])
        initcond[s] = x[i, j]
    end
end

modelparam = OrderedDict()
np = sum(tunable_parameters(neuronmodel); init=0) do par
    val = Symbolics.getdefaultval(par)
    modelparam[par] = val
    length(val)
end
indices = Dict(:dspars => collect(1:np))
# Noise parameter mean
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
n = length(modelparam[:lnα]);
indices[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
indices[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = zeros(Float64, nrr);   # region specific observation noise
indices[:lnγ] = collect(np+1:np+nrr);
np += nrr
indices[:u] = idx_u
indices[:m] = idx_bold
indices[:sts] = idx_sts
# define prior variances
paramvariance = copy(modelparam)
paramvariance[:lnγ] = ones(Float64, nrr)./64.0;
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0;
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
for (k, v) in paramvariance
    if occursin("A", string(k))
        paramvariance[k] = ones(length(v))
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperpriors = (Πλ_pr = vars["ihC"]*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
               μλ_pr = [vars["hE"]]              # prior metaparameter mean, needs to be a vector
               );

csdsetup = (p = 8, freq = vec(vars["Hz"]), dt = vars["dt"]);

(state, setup) = setup_sDCM(data, neuronmodel, initcond, csdsetup, priors, hyperpriors, indices);

# HACK: on machines with very small amounts of RAM, Julia can run out of stack space while compiling the code called in this loop
# this should be rewritten to abuse the compiler less, but for now, an easy solution is just to run it with more allocated stack space.
with_stack(f, n) = fetch(schedule(Task(f,n)))

with_stack(5_000_000) do  # 5MB of stack space
    for iter in 1:max_iter
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
end
print("maxixmum iterations reached\n")

### COMPARE RESULTS WITH MATLAB RESULTS ###
@show state.F[end], vars["F"]
@test state.F[end] < vars["F"]*0.99
@test state.F[end] > vars["F"]*1.01
