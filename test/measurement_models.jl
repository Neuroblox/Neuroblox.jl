using Neuroblox, Test

using MAT
using LinearAlgebra: I

matvars = matread(joinpath(@__DIR__, "DEM_demo_induced_fMRI_SPM12.mat"));
x = vcat(zeros(1,15), Matrix(matvars["x"]));
transit = vec(matvars["transit"]);
decay = 0.0
dt = 2
ns = size(x, 1)    # number of samples
nd = size(x, 2)    # total number of dynamic variables
nr = nd ÷ 5        # number of brain regions

dx = zeros(nr, 4, ns);
J = zeros(nd-nr, nd-nr, ns);
for i = 1:ns
    dx[:,:,i], J[:,:,i] = Hemodynamics!(reshape(x[i,(nr+1):end],nr,4), x[i, 1:nr], decay, transit)
end

@test matvars["dfdx256"][nr+1:end,nr+1:end] ≈ J[:,:,256]
@test matvars["dfdx512"][nr+1:end,nr+1:end] ≈ J[:,:,512]
@test matvars["fx256"][nr+1:end] ≈ reshape(dx[:,:,256],nd-nr)
@test matvars["fx512"][nr+1:end] ≈ reshape(dx[:,:,512],nd-nr)


lnϵ = 0
x = x[2:end, :]
ns = size(x,1)
fMRI = zeros(ns,nr);
gradient = zeros(ns, nr, 2nr);
for i = 1:ns
    fMRI[i, :], gradient[i, :, :] = boldsignal(x[i, (3nr+1):4nr], x[i, (4nr+1):end], lnϵ);
end

@test matvars["gradient256"] ≈ gradient[256,:,:]
@test matvars["gradient512"] ≈ gradient[512,:,:]
@test matvars["y"] ≈ fMRI
