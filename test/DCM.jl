using Neuroblox, Test

using MAT

### DEFINE SEVERAL VARIABLES AND PRIORS TO GET STARTED ###

vars = matread("test/spectralDCM_demodata.mat");
y_csd = vars["csd"];
w = vec(vars["M_nosparse"]["Hz"]);
A = vars["M_nosparse"]["pE"]["A"];    # see table 1 in friston2014 for values of priors 
θΣ = vars["M_nosparse"]["pC"];
λμ = vec(vars["M_nosparse"]["hE"]);
Πλ_p = vars["M_nosparse"]["ihC"];

idx = findall(x -> x != 0, θΣ);
V = zeros(size(θΣ, 1), length(idx));
order = sortperm(θΣ[idx], rev=true);
idx = idx[order];
for i = 1:length(idx)
    V[idx[i][1], i] = 1.0
end
θΣ = V'*θΣ*V;
Πθ_p = inv(θΣ);

dim = size(A, 1);
C = zeros(Float64, dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
p = 8;
α = [0.0, 0.0];
β = [0.0, 0.0];
γ = zeros(Float64, dim);
lnϵ = 0.0;                        # BOLD signal parameter
lndecay = 0.0;                    # hemodynamic parameter
lntransit = zeros(Float64, dim);  # hemodynamic parameters
x = zeros(Float64, dim, 5);       # initial states of dynamical equation system

param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ;]
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
priors = [Πθ_p, Πλ_p, λμ]
niter = 128


### ESTIMATE DYNAMIC CAUSAL MODEL ###
results = VariationalBayes(x, y_csd, w, V, param, priors, niter)

### COMPARE RESULTS WITH MATLAB RESULTS ###
@test_broken results["F"] ≈ vars["F"]