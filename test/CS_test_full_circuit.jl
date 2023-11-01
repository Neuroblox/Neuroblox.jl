using Neuroblox
using DifferentialEquations
using MetaGraphs



    global_ns = :g 
    @named LC = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) #1
	@named ITN = NextGenerationEIBlox(;namespace=global_ns, Cₑ=2*36,Cᵢ=1*36, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/36, alpha_invₑᵢ=0.8/36, alpha_invᵢₑ=10.0/36, alpha_invᵢᵢ=0.8/36, kₑₑ=0.0*36, kₑᵢ=0.6*36, kᵢₑ=0.6*36, kᵢᵢ=0*36) #2

	@named VC = CorticalBlox(N_wta=45, N_exci=5,  density=0.015, weight=1,I_bg_ar=20;namespace=global_ns) #3
    @named PFC = CorticalBlox(N_wta=20, N_exci=5, density=0.015, weight=1,I_bg_ar=0;namespace=global_ns) #4

	@named STR1 = Striatum(N_inhib=25;namespace=global_ns) #5
	@named STR2 = Striatum(N_inhib=25;namespace=global_ns) #6
	@named tan_nrn = HHNeuronExciBlox(;namespace=global_ns) #7
    @named gpi1 = GPi(N_inhib=25;namespace=global_ns) #8
	@named gpi2 = GPi(N_inhib=25;namespace=global_ns) #9
	@named gpe1 = GPe(N_inhib=25;namespace=global_ns) #10
	@named gpe2 = GPe(N_inhib=25;namespace=global_ns) #11
	@named STN1 = STN(N_exci=25,I_bg=3*ones(25);namespace=global_ns) #12
    @named STN2 = STN(N_exci=25,I_bg=3*ones(25);namespace=global_ns) #13
	@named Thal1 = Thalamus(N_exci=25;namespace=global_ns) #14
	@named Thal2 = Thalamus(N_exci=25;namespace=global_ns) #15
	
    assembly = [LC, ITN, VC, PFC, STR1, STR2, tan_nrn, gpi1, gpi2, gpe1, gpe2, STN1, STN2, Thal1, Thal2]

	g = MetaDiGraph()
    add_blox!.(Ref(g), assembly)

	add_edge!(g,1,3, Dict(:weight => 44)) #LC->VC
	add_edge!(g,1,4, Dict(:weight => 44)) #LC->pfc
	add_edge!(g,2,7, Dict(:weight => 100)) #ITN->tan
	add_edge!(g,3,4, Dict(:weight => 3, :density => 0.08)) #VC->pfc
	add_edge!(g,4,5, Dict(:weight => 0.06, :density => 0.04)) #pfc->str1
	add_edge!(g,4,6, Dict(:weight => 0.06, :density => 0.04)) #pfc->str2
	add_edge!(g,7,5, Dict(:weight => 0.17)) #tan->str1
	add_edge!(g,7,6, Dict(:weight => 0.17)) #tan->st2
	add_edge!(g,5,8, Dict(:weight => 4, :density => 0.04)) #str1->gpi1
	add_edge!(g,6,9, Dict(:weight => 4, :density => 0.04)) #str2->gpi2
	add_edge!(g,8,14, Dict(:weight => 0.16, :density => 0.04)) #gpi1->thal1
	add_edge!(g,8,14, Dict(:weight => 0.16, :density => 0.04)) #gpi2->thal2
    add_edge!(g,14,4, Dict(:weight => 1, :density => 0.32)) #thal1->pfc
	add_edge!(g,15,4, Dict(:weight => 1, :density => 0.32)) #thal2->pfc
	add_edge!(g,5,10, Dict(:weight => 4, :density => 0.04)) #str1->gpe1
	add_edge!(g,6,11, Dict(:weight => 4, :density => 0.04)) #str2->gpe2
	add_edge!(g,10,8, Dict(:weight => 0.2, :density => 0.04)) #gpe1->gpi1
	add_edge!(g,11,9, Dict(:weight => 0.2, :density => 0.04)) #gpe2->gpi2
	add_edge!(g,10,12, Dict(:weight => 3.5, :density => 0.04)) #gpe1->stn1
	add_edge!(g,11,13, Dict(:weight => 3.5, :density => 0.04)) #gpe2->stn2
	add_edge!(g,12,8, Dict(:weight => 0.1, :density => 0.04)) #gpe1->stn1
	add_edge!(g,13,9, Dict(:weight => 0.1, :density => 0.04)) #gpe2->stn2

    neuron_net = system_from_graph(g; name=global_ns);
    prob = ODEProblem(structural_simplify(neuron_net), [], (0.0, 1000), []);
    sol = solve(prob,QNDF(),saveat=0.01)