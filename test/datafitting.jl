using Neuroblox, Test, Graphs, MetaGraphs, OrderedCollections, LinearAlgebra, DataFrames
using MAT


@testset "fMRI test" begin
    ### Load data ###
    vars = matread(joinpath(@__DIR__, "spm25_fMRI_toydata.mat"));
    data = DataFrame(vars["data"], :auto)    # turn data into DataFrame, name column names after building the model.
    x = vars["x"]                            # point around which expansion is computed
    nrr = ncol(data)                         # number of recorded regions
    ns  = nrow(data)                         # number of samples
    max_iter = 128
    dt = 2.0                                 # time bin in seconds
    freq = range(min(128, ns*dt)^-1, max(8, 2*dt)^-1, 32)  # define frequencies at which to evaluate the CSD

    ########## assemble the model ##########
    g = MetaDiGraph()
    regions = Dict()
    @parameters lnκ=0.0 [tunable = true] lnϵ=0.0 [tunable=true] C=1/16 [tunable = false]
    for ii = 1:nrr
        region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
        add_blox!(g, region)
        regions[ii] = nv(g)    # store index of neural mass model
        taskinput = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
        add_edge!(g, taskinput => region, weight = C)
        # add hemodynamic observer
        observer = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=lnκ, lnϵ=lnϵ)
        # connect observer with neuronal signal
        add_edge!(g, region => observer, weight = 1.0)
    end

    # add symbolic weights
    A = []
    for (i, a) in enumerate(vec(vars["pE"]["A"]))
        symb = Symbol("A$(i)")
        push!(A, only(@parameters $symb = a))
    end

    for (i, idx) in enumerate(CartesianIndices(vars["pE"]["A"]))
        if idx[1] == idx[2]
            add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, -exp(A[i])/2)  # treatement of diagonal elements in SPM, likely to avoid instabilities of the linear model
        else
            add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
        end
    end

    # compose model
    @named neuronmodel = system_from_graph(g, simplify=false)
    untunelist = Dict()  # dictionary of parameters whose tunable flag may be changed, we do this in dependency of variances that are set to 0 as is done in SPM
    for (i, v) in enumerate(diag(vars["pC"])[1:nrr^2])
        untunelist[A[i]] = v == 0 ? false : true
    end
    neuronmodel = changetune(neuronmodel, untunelist)
    neuronmodel = structural_simplify(neuronmodel)

    # attribute initial conditions to states
    _, obsvars = get_eqidx_tagged_vars(neuronmodel, "measurement")  # get index of equation of bold state
    rename!(data, Symbol.(obsvars))

    sts, _ = get_dynamic_states(neuronmodel)
    initcond = OrderedDict(sts .=> 0.0)
    rnames = []
    map(x->push!(rnames, split(string(x), "₊")[1]), sts);
    rnames = unique(rnames);
    for (i, r) in enumerate(rnames)
        for (j, s) in enumerate(sts[r .== map(x -> x[1], split.(string.(sts), "₊"))])
            initcond[s] = x[i, j]
        end
    end

    pmean, pcovariance, indices = defaultprior(neuronmodel, nrr)
    # priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
    priors = (μθ_pr = pmean,
              Σθ_pr = pcovariance
    );
    hyperpriors = (Πλ_pr = 128.0*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   μλ_pr = [8.0]               # prior metaparameter mean, needs to be a vector
                );

    csdsetup = (mar_order = 8, freq = freq, dt = dt);

    (state, setup) = setup_sDCM(data, neuronmodel, initcond, csdsetup, priors, hyperpriors, indices, pmean, "fMRI");

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

    ### COMPARE RESULTS WITH MATLAB RESULTS ###
    @show state.F[end], vars["F"]
    @test state.F[end] < vars["F"]*0.99
    @test state.F[end] > vars["F"]*1.01
end

@testset "LFP test" begin
    ### Load data ###
    vars = matread(joinpath(@__DIR__, "spm12_cmc.mat"));
    data = DataFrame(vars["data"], :auto)    # turn data into DataFrame, name column names after building the model.
    x = vars["x"]                            # point around which expansion is computed
    nrr = ncol(data)                         # number of recorded regions
    ns  = nrow(data)                         # number of samples
    max_iter = 128
    dt = 2.0                                 # time bin in seconds
    freq = range(1.0, 64.0)  # define frequencies at which to evaluate the CSD

    ########## assemble the model ##########
    g = MetaDiGraph()
    global_ns = :g                            # global namespace
    regions = Dict()

    @parameters lnr = 0.0
    @parameters lnτ_ss=0 lnτ_sp=0 lnτ_ii=0 lnτ_dp=0
    @parameters C=512.0 [tunable = false]    # TODO: SPM has this seemingly arbitrary 512 pre-factor in spm_fx_cmc.m. Can we understand why?
    for ii = 1:nrr
        region = CanonicalMicroCircuitBlox(;namespace=global_ns, name=Symbol("r$(ii)₊cmc"), 
                                            τ_ss=exp(lnτ_ss)*0.002, τ_sp=exp(lnτ_sp)*0.002, τ_ii=exp(lnτ_ii)*0.016, τ_dp=exp(lnτ_dp)*0.028, 
                                            r_ss=exp(lnr)*2.0/3, r_sp=exp(lnr)*2.0/3, r_ii=exp(lnr)*2.0/3, r_dp=exp(lnr)*2.0/3)
        add_blox!(g, region)
        regions[ii] = nv(g)    # store index of neural mass model
        input = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
        add_edge!(g, input => region; weight = C)

        # add lead field (LFP measurement)
        measurement = LeadField(;name=Symbol("r$(ii)₊lf"))
        # connect measurement with neuronal signal
        add_edge!(g, region => measurement; weight = 1.0)
    end

    nl = Int((nrr^2-nrr)/2)   # number of links unidirectional
    @parameters a_sp_ss[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> ss
    @parameters a_sp_dp[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> dp
    @parameters a_dp_sp[1:nl] = repeat([0.0], nl) # backward connection parameter dp -> sp
    @parameters a_dp_ii[1:nl] = repeat([0.0], nl) # backward connection parameter dp -> ii

    k = 0
    for i in 1:nrr
        for j in (i+1):nrr
            k += 1
            # forward connection matrix
            add_edge!(g, regions[i], regions[j], :weightmatrix,
                        [0 exp(a_sp_ss[k]) 0 0;            # connection from sp to ss
                        0 0 0 0;
                        0 0 0 0;
                        0 exp(a_sp_dp[k])/2 0 0] * 200)    # connection from sp to dp
            # backward connection matrix
            add_edge!(g, regions[j], regions[i], :weightmatrix,
                        [0 0 0 0;
                        0 0 0 -exp(a_dp_sp[k]);            # connection from dp to sp
                        0 0 0 -exp(a_dp_ii[k])/2;          # connection from dp to ii
                        0 0 0 0] * 200)
        end
    end

    @named fullmodel = system_from_graph(g)

    # attribute initial conditions to states
    sts, idx_sts = get_dynamic_states(fullmodel)
    idx_u = get_idx_tagged_vars(fullmodel, "ext_input")                # get index of external input state
    idx_measurement, obsvars = get_eqidx_tagged_vars(fullmodel, "measurement")  # get index of equation of bold state
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
    np = sum(tunable_parameters(fullmodel); init=0) do par
        val = Symbolics.getdefaultval(par)
        modelparam[par] = val
        length(val)
    end
    indices = Dict(:dspars => collect(1:np))
    # Noise parameter mean
    modelparam[:lnα] = zeros(Float64, 2, nrr);        # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
    n = length(modelparam[:lnα]);
    indices[:lnα] = collect(np+1:np+n);
    np += n;
    modelparam[:lnβ] = [-16.0, -16.0];                # global observation noise, ln(β) as above
    n = length(modelparam[:lnβ]);
    indices[:lnβ] = collect(np+1:np+n);
    np += n;
    modelparam[:lnγ] = [-16.0, -16.0];                # region specific observation noise
    indices[:lnγ] = collect(np+1:np+nrr);
    np += nrr
    indices[:u] = idx_u
    indices[:m] = idx_measurement
    indices[:sts] = idx_sts

    # define prior variances
    paramvariance = copy(modelparam)
    paramvariance[:lnα] = ones(Float64, size(modelparam[:lnα]))./128.0; 
    paramvariance[:lnβ] = ones(Float64, nrr)./128.0;
    paramvariance[:lnγ] = ones(Float64, nrr)./128.0;
    for (k, v) in paramvariance
        if occursin("a_", string(k))
            paramvariance[k] = 1/16.0
        elseif "lnr" == string(k)
            paramvariance[k] = 1/64.0;
        elseif occursin("lnτ", string(k))
            paramvariance[k] = 1/32.0;
        elseif occursin("lf₊L", string(k))
            paramvariance[k] = 64;
        end
    end

    # priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
    priors = (μθ_pr = modelparam,
                Σθ_pr = paramvariance
                );

    hype = matread(joinpath(@__DIR__, "spm12_cmc_hyperpriors.mat"));
    hyperpriors = (Πλ_pr = hype["ihC"],               # prior metaparameter precision, needs to be a matrix
                   μλ_pr = vec(hype["hE"]),           # prior metaparameter mean, needs to be a vector
                   Q = hype["Q"]);

    csdsetup = (mar_order = 8, freq = freq, dt = dt);

    (state, setup) = setup_sDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, modelparam, "LFP");

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

    ### COMPARE RESULTS WITH MATLAB RESULTS ###
    @show state.F[end]
    @test state.F[end] > 1891*0.99
    @test state.F[end] < 1891*1.01
end
