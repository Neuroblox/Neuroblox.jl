using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

#Number of neurons in each block
N_pyr=80
N_in_phasic=20
N_in_tonic=80

#number of connections between each block pair
prj_pyr_phasic = 10
prj_phasic_pyr = 10
prj_tonic_pyr = 5
prj_tonic_phasic = 5

@named cb_pyr = Cortical_Pyramidal_Assembly_Adam(N_exci=N_pyr,namespace=:g)
@named cb_in_phasic = Cortical_Interneuron_Assembly_Phasic_Adam(namespace=:g,N_inhib=N_in_phasic)
@named cb_in_tonic = Cortical_Interneuron_Assembly_Tonic_Adam(namespace=:g,N_inhib=N_in_tonic)
@named glu= Steady_Glutamate()
assembly = [cb_pyr,cb_in_phasic,cb_in_tonic,glu2]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)

add_edge!(g,1,2,Dict(:weight=>0.2/prj_pyr_phasic,:density=>prj_pyr_phasic/N_pyr,:nmda=>true,:nmda_weight=>9.5/prj_pyr_phasic))
add_edge!(g,2,1,Dict(:weight=>0.8/prj_phasic_pyr,:density=>prj_phasic_pyr/N_in_phasic))
add_edge!(g,3,1,Dict(:weight=>5/prj_tonic_pyr,:density=>prj_tonic_pyr/N_in_tonic))
add_edge!(g,3,2,Dict(:weight=>5/prj_tonic_phasic,:density=>prj_tonic_phasic/N_in_tonic))

@named neuron_net = system_from_graph(g)
sys = structural_simplify(neuron_net)
prob = SDEProblem(sys, [], (0.0, 2000), [])
sol = solve(prob, ImplicitEM(),saveat = 0.01)
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
plot(sol.t,VV[:,:]',color= "blue",label=false)


