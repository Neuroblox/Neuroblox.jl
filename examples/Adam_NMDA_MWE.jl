using Neuroblox
using DifferentialEquations
using Graphs
using MetaGraphs
using Plots

@named n1=HHNeuronExci_pyr_Adam_Blox(σ=0,I_bg=-1.25;N_nmda=1,namespace=:g)
@named n2=HHNeuronExci_pyr_Adam_Blox(σ=3;N_nmda=1,namespace=:g)
#this parameter set reproduces figure 3C    DO NOT CHANGE
@named n3=HHNeuronInh_inter_Adam_Blox(σ=1,I_bg=-1.8;N_nmda=1,k_unblock=5.4,namespace=:g)

@named glu1= Glutamate_puff()
@named glu2= Steady_Glutamate()

assembly = [n1,n2]
#assembly = [glu1,n1]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g,1,2,Dict(:weight=>0,:nmda=>true,:nmda_weight=>2))

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