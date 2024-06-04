using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots
using Statistics
using DSP

global_ns = :g
@named msn = Striatum_MSN_Adam(namespace=global_ns)
@named fsi = Striatum_FSI_Adam(namespace=global_ns)
@named gpe = GPe_Adam(namespace=global_ns)
@named stn = STN_Adam(namespace=global_ns)

assembly = [msn, fsi, gpe, stn]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)

add_edge!(g, 1, 3, Dict(:weight=> 2.5/33, :density=>0.33))
add_edge!(g, 2, 1, Dict(:weight=> 0.6/7.5, :density=>0.15))
add_edge!(g, 3, 4, Dict(:weight=> 0.3/4, :density=>0.05))
add_edge!(g, 4, 2, Dict(:weight=> 0.165/4, :density=>0.1))

@named neuron_net = system_from_graph(g)
sys = structural_simplify(neuron_net)
prob = SDEProblem(sys, [], (0.0, 500), [])
sol = solve(prob, saveat = 0.01)
ss=convert(Array,sol)

st=unknowns(sys)
vlist=Int64[]
for ii = 1:length(st)
    if contains(string(st[ii]), "V(t)")
            push!(vlist,ii)
    end
end
V = ss[vlist,:]
	
VV=zeros(length(vlist),length(sol.t))
for ii = 1:length(vlist)
    VV[ii,:] .= V[ii,:] .+ 200*(ii-1)

end

mmsn=mean(V[1:100,:],dims=1)
mfsi=mean(V[101:150,:],dims=1)
mgpe=mean(V[151:230,:],dims=1)
mstn=mean(V[231:end,:],dims=1)

V_av=mmsn

fs = 1000;
avec = [ii*100+1 for ii = 1:500]
V_av = mean(V,dims=1)
periodogram_estimation = periodogram(V_av[1,avec], fs=fs)
#periodogram_estimation = welch_pgram(average1[1,avec], fs=fs)
pxx = periodogram_estimation.power
f = periodogram_estimation.freq

plot!(f,log10.(pxx),xlabel="frequency Hz",ylabel="log(psd)",xlims=(0,150))