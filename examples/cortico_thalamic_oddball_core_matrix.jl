### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 170d98f5-1177-4d97-9cb3-c899c5aa5f41
begin
using Pkg
Pkg.activate(".")
end

# ╔═╡ 0d29fc66-498d-4637-a3e2-ed16ab1d8bae
@time using Neuroblox

begin

    Npulse = 5
    ptime_mat=zeros(5,Npulse);
    ptime_mat[1,:]=collect(100:1250:1250*Npulse);
    ptime_mat[2,:]=ptime_mat[1,:].+150;
    ptime_mat[3,:]=ptime_mat[2,:].+150;
    ptime_mat[4,:]=ptime_mat[3,:].+150;
    ptime_mat[5,:]=ptime_mat[4,:].+150;
    ptime = reshape(ptime_mat,5*Npulse,1);
    
    pswitch_r = ones(5*Npulse,1);
    pswitch_o_1 = ones(5*Npulse,1);
    for ii=5:5:5*Npulse
        pswitch_o_1[ii,1]=0;
    end
    
    pswitch_o_2 = zeros(5*Npulse,1);
    for ii=5:5:5*Npulse
        pswitch_o_2[ii,1]=1;
    end
    
    tspan = ptime[end] +300
end
# ╔═╡ 2ea6657c-0336-11f0-256b-0d2ed24ac5e6
begin
	
	
	using OrdinaryDiffEq 
	using Random
	using CairoMakie
	#using Plots
	using Statistics
    using DelimitedFiles
	
end
"""
begin
    global_namespace=:g;
    @named c1= CorticalBlox(N_wta=1,I_bg_ar=0; namespace=global_namespace);
    @named c2= CorticalBlox(N_wta=1,I_bg_ar=0; namespace=global_namespace);
    wm=[3 0.5 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0 ];
    g = MetaDiGraph();
    add_edge!(g,c1=>c2, weightmatrix = wm)
    @time sys = system_from_graph(g, name=global_namespace);
    defs = ModelingToolkit.get_defaults(sys);
    ps=parameters(sys);
    id1=findall(x -> occursin("I_bg", String(Symbol(x))), ps);
    id2=findall(x -> occursin("exci", String(Symbol(x))), ps);
    id3=findall(x -> occursin("c1", String(Symbol(x))), ps);
    id3=intersect(id1,id2,id3);
    
    defs[ps[id3[1]]]=3
    pv=ModelingToolkit.MTKParameters(sys,defs);

    prob = ODEProblem(sys, [], (0.0, 2000), []);
    @time sol = solve(prob, Vern7(), saveat=0.1);    


end
"""
# ╔═╡ 7c04d938-c736-428c-923f-fc1e4c9a46bb
begin
	global_namespace=:g;

    gaba_param=1

    @named ASC1 = NextGenerationEIBlox(;namespace=global_namespace, Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 

    @named Layer_4_A = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=2, I_bg_ar=0, G_syn_inhib=0.2,G_syn_exci=0.5,G_syn_ff_inhib=1.5, τ_inhib=70*gaba_param; namespace=global_namespace);
    #@named Layer_4_A = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=5, I_bg_ar=0, G_syn_inhib=0.6,G_syn_exci=0.8; namespace=global_namespace);
    
    #@named Layer_6_A = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=5, I_bg_ar=0, G_syn_inhib=0.6, G_syn_exci=0.8; namespace=global_namespace);
    @named Layer_6_A = CorticalBlox(N_wta=5, N_exci=5, density=0.05, weight=0.5, I_bg_ar=0, G_syn_inhib=0.0, G_syn_exci=0.7, τ_inhib=70*gaba_param, G_syn_ff_inhib=1.5; namespace=global_namespace);

    @named Layer_2_3_A = CorticalBlox(N_wta=10, N_exci=5, density=0.01, weight=1, I_bg_ar=4, G_syn_inhib=4, τ_inhib=70*gaba_param, G_syn_ff_inhib=1.5,; namespace=global_namespace);

    @named Layer_5_A = CorticalBlox(N_wta=5, N_exci=5, density=0.05, weight=3, I_bg_ar=0, G_syn_inhib=0.2,G_syn_exci=0.5,G_syn_ff_inhib=1.5, τ_inhib=70*gaba_param; namespace=global_namespace);#τ₃=100
    
    @named Thal_mat_A =  Thalamus(N_exci=25,I_bg=zeros(25),density=0.03,weight=1.3;namespace=global_namespace);#0.025 
    



    @named Thal_core_A =  Thalamus(N_exci=25,I_bg=-2.5 .+ (2.75)*rand(25),density=0.03,weight=1.3;namespace=global_namespace);#0.025 
    @named Nrt_A =  GPi(N_inhib=25, I_bg = zeros(25), τ_inhib=250*gaba_param, G_syn_inhib=8;namespace=global_namespace) ;

	@named InfC_A = PulsesInput(pulse_amp=0, pulse_switch = [1 1 1 1 0] , pulse_width=50, t_start= 00 .+ [100 250 400 550 700] ;namespace=global_namespace);#4
    #@named InfC_A_ = PulsesInput(pulse_amp=0, pulse_switch = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] , pulse_width=50, t_start= 00 .+ [100 250 400 550 700 1350 1500 1650 1800 1950 2600 2750 2900 3050 3200 3850 4000 4150 4300 4450] ;namespace=global_namespace);#4
    @named InfC_A_r = PulsesInput(pulse_amp=0, pulse_switch = pswitch_r, pulse_width=50, t_start= ptime ;namespace=global_namespace);#4
    @named InfC_A_o = PulsesInput(pulse_amp=0, pulse_switch = pswitch_o_1, pulse_width=50, t_start= ptime ;namespace=global_namespace);#4

 
 
 
    @named Layer_4_B = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=2, I_bg_ar=0, G_syn_inhib=0.2,G_syn_exci=0.5,G_syn_ff_inhib=1.5, τ_inhib=70*gaba_param; namespace=global_namespace);
    #@named Layer_4_A = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=5, I_bg_ar=0, G_syn_inhib=0.6,G_syn_exci=0.8; namespace=global_namespace);
    
    #@named Layer_6_A = CorticalBlox(N_wta=5, N_exci=5, density=0.03, weight=5, I_bg_ar=0, G_syn_inhib=0.6, G_syn_exci=0.8; namespace=global_namespace);
    @named Layer_6_B = CorticalBlox(N_wta=5, N_exci=5, density=0.05, weight=0.5, I_bg_ar=0, G_syn_inhib=0.0, G_syn_exci=0.7, τ_inhib=70*gaba_param, G_syn_ff_inhib=1.5; namespace=global_namespace);
    @named Layer_2_3_B = CorticalBlox(N_wta=10, N_exci=5, density=0.01, weight=1, I_bg_ar=4, G_syn_inhib=4, τ_inhib=70*gaba_param, G_syn_ff_inhib=1.5; namespace=global_namespace);


    @named Thal_core_B =  Thalamus(N_exci=25,I_bg=-2.5 .+ (2.75)*rand(25),density=0.03,weight=1.3;namespace=global_namespace);#0.025 
    @named Nrt_B =  GPi(N_inhib=25, I_bg = zeros(25), τ_inhib=250*gaba_param, G_syn_inhib=8;namespace=global_namespace) ;

	@named InfC_B = PulsesInput(pulse_amp=0, pulse_switch = [0 0 0 0 1] , pulse_width=50, t_start= 00 .+ [100 250 400 550 700] ;namespace=global_namespace);#4
    #@named InfC_B_ = PulsesInput(pulse_amp=0, pulse_switch = [0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1] , pulse_width=50, t_start= 00 .+ [100 250 400 550 700 1350 1500 1650 1800 1950 2600 2750 2900 3050 3200 3850 4000 4150 4300 4450] ;namespace=global_namespace);#4
    @named InfC_B_o = PulsesInput(pulse_amp=0, pulse_switch = pswitch_o_2, pulse_width=50, t_start= ptime ;namespace=global_namespace);#4

    @named Thal_mat_B =  Thalamus(N_exci=25,I_bg=zeros(25),density=0.03,weight=1.3;namespace=global_namespace);#0.025 
    @named Layer_5_B = CorticalBlox(N_wta=5, N_exci=5, density=0.05, weight=3, I_bg_ar=0, G_syn_inhib=0.2,G_syn_exci=0.5,G_syn_ff_inhib=1.5, τ_inhib=70*gaba_param; namespace=global_namespace);#τ₃=100
    @named Layer_2_3_ac = CorticalBlox(N_wta=10, N_exci=5, density=0.001, weight=1, I_bg_ar=0, G_syn_inhib=12, G_syn_exci=6.0, τ_inhib=100*gaba_param, G_syn_ff_inhib=1.5,; namespace=global_namespace);
   
     

end

#connection matrices
begin

    CC_mat_A =zeros(50,50)
    CC_mat_B =zeros(50,50)

    TC_mat_A =zeros(25,50)
    TC_mat_B =zeros(25,50)

    wt_CC_A=0.8;
    wt_CC_B=0.8;
    p_CC=0.7

    CC_indeg=20

    wt_TC_A=250;
    wt_TC_B=200;
    wt_TC_=0.1;
    p_TC=0.3;

    for jj in 1:50
        if rand() < p_CC
            src = randperm(50)
            CC_mat_A[src[1:CC_indeg],jj] .=  wt_CC_A
        end

        if rand() < p_TC
            src = rand(1:25)
            if maximum(CC_mat_A[:,jj]) > 0
                TC_mat_A[src,jj] =  wt_TC_A
            else
                TC_mat_A[src,jj] =  wt_TC_
            end
            
        end


        if rand() < p_CC
            src = randperm(50)
            CC_mat_B[src[1:CC_indeg],jj] .=  wt_CC_B
        end

        if rand() < p_TC
            src = rand(1:25)
            if maximum(CC_mat_B[:,jj]) > 0
                TC_mat_B[src,jj] =  wt_TC_B
            else
                TC_mat_B[src,jj] =  wt_TC_
            end
            
        end

    end
end
# ╔═╡ 90f96021-80e1-4cd8-b5e7-9818c2b6c21f
begin

g = MetaDiGraph();
"""
add_blox!(g,ASC1);
add_blox!(g,Layer_2_3_A);
add_blox!(g,Layer_4_A);
add_blox!(g,Layer_6_A);
add_blox!(g,InfC_A);
add_blox!(g,Thal_core_A);   
add_blox!(g,Nrt_A);


add_blox!(g,Layer_2_3_B);
add_blox!(g,Layer_4_B);
add_blox!(g,Layer_6_B);
add_blox!(g,InfC_B);
add_blox!(g,Thal_core_B);
add_blox!(g,Nrt_B);
"""

add_edge!(g,ASC1 => Layer_2_3_A, weight = 20);
add_edge!(g,ASC1 => Layer_4_A, weight = 20);
add_edge!(g,ASC1 => Layer_5_A, weight = 20);
add_edge!(g,InfC_A_o => Thal_core_A, weight = 0.0);
add_edge!(g,InfC_A_r => Thal_core_A, weight = 0.1);
add_edge!(g, Thal_core_A => Layer_6_A, weight = 1, density = 0.04);
#add_edge!(g, Thal_core_A => Layer_4_A, weight = 2, density = 0.2);
add_edge!(g, Thal_core_A => Layer_4_A, weight = 5, density = 0.04);
#add_edge!(g, Layer_6_A => Nrt_A, weight = 0.6, density = 0.3);
add_edge!(g, Layer_6_A => Nrt_A, weight = 4, density = 0.04);
#add_edge!(g, Nrt_A => Thal_core_A, weight = 1.1e-4, density = 0.7);
add_edge!(g, Nrt_A => Thal_core_A, weight = 4e-3, density = 0.4);
add_edge!(g, Layer_6_A => Thal_core_A, weight = 0.4, density = 0.3);
add_edge!(g, Layer_4_A => Layer_2_3_A, weight = 4, density = 0.08);
add_edge!(g, Layer_2_3_A => Layer_6_A, weight = 0.4, density = 0.05);#w=0.6

#matrix
add_edge!(g, Layer_2_3_A => Layer_5_A, weight = 0.08, density = 0.8);#0.2
add_edge!(g, Layer_5_A => Thal_mat_A, weight = 0.6, density = 0.4);#0.5
add_edge!(g, Thal_mat_A => Layer_5_A, weight = 1, density = 0.04, sta=true);#1
add_edge!(g, Thal_mat_A => Layer_2_3_A, weight = 0, density = 0.04, sta=true);#0.5


add_edge!(g,ASC1 => Layer_2_3_B, weight = 20);
add_edge!(g,ASC1 => Layer_4_B, weight = 20);
add_edge!(g,ASC1 => Layer_5_B, weight = 20);
add_edge!(g,InfC_B_o => Thal_core_B, weight = 0.0);
#add_edge!(g,InfC_B_ => Thal_core_B, weight = 0.1);
add_edge!(g, Thal_core_B => Layer_6_B, weight = 1, density = 0.04);
add_edge!(g, Thal_core_B => Layer_4_B, weight = 5, density = 0.04);
add_edge!(g, Layer_6_B => Nrt_B, weight = 4, density = 0.04);#w=4
add_edge!(g, Nrt_B => Thal_core_B, weight = 4e-3, density = 0.4);#w=1.4e-2
add_edge!(g, Layer_6_B => Thal_core_B, weight = 0.4, density = 0.3);
add_edge!(g, Layer_4_B => Layer_2_3_B, weight = 4, density = 0.08);
add_edge!(g, Layer_2_3_B => Layer_6_B, weight = 0.4, density = 0.05);	

#matrix
add_edge!(g, Layer_2_3_B => Layer_5_B, weight = 0.08, density = 0.8);#0.2
add_edge!(g, Layer_5_B => Thal_mat_B, weight = 0.6, density = 0.4);#0.5
add_edge!(g, Thal_mat_B => Layer_5_B, weight = 1, density = 0.04, sta=true);#1
add_edge!(g, Thal_mat_B => Layer_2_3_B, weight = 0, density = 0.04, sta=true);#0.5

# anterior cortex 
add_edge!(g, Layer_2_3_A => Layer_2_3_ac, weightmatrix = CC_mat_A);#0.2
add_edge!(g, Thal_mat_A => Layer_2_3_ac, weightmatrix = TC_mat_A, sta=true);#0.2
add_edge!(g, Layer_2_3_B => Layer_2_3_ac, weightmatrix = CC_mat_B);#0.2
add_edge!(g, Thal_mat_B => Layer_2_3_ac, weightmatrix = TC_mat_B, sta=true);#0.2

@time sys = system_from_graph(g, name=global_namespace);	


end

begin
    prob = ODEProblem(sys, [], (0.0, 2000), []);
    @time sol_ = solve(prob, Vern7(), saveat=0.1);
    u0 = sol_[:,end];
    un = unknowns(sys);
    id = findall(x -> occursin("Gₛₜₚ", String(Symbol(x))), un);
    u0[id] .= 0;

    ps=parameters(sys);
    defs = ModelingToolkit.get_defaults(sys);

    # GABAergic synapse timescale.
    id_in1=findall(x -> occursin("τ₂", String(Symbol(x))), ps);
    id_in2=findall(x -> occursin("inh", String(Symbol(x))), ps);
    id_in3=intersect(id_in1,id_in2);
    inh_t=ps[id_in3];
    init_ts=zeros(length(inh_t));
    for i in 1:length(inh_t)
        init_ts[i] = copy(defs[inh_t[i]]);
    end
    
    # GABAergic synapse assymptotic current.
    id_in_g1=findall(x -> occursin("G_syn", String(Symbol(x))), ps);
    id_in_g2=findall(x -> occursin("inh", String(Symbol(x))), ps);
    id_in_g3=intersect(id_in_g1,id_in_g2);
    inh_g=ps[id_in_g3];
    init_g=zeros(length(inh_g));
    for i in 1:length(inh_g)
        init_g[i] = copy(defs[inh_g[i]]);
    end

    # background currents for thal
    
    id1_t=findall(x -> occursin("I_bg", String(Symbol(x))), ps);
    id2_t=findall(x -> occursin("Thal", String(Symbol(x))), ps);
    id_bg_thal=intersect(id1_t,id2_t);
        
    I_thal = ps[id_bg_thal];
    
    I_init_thal=zeros(length(I_thal));
    for i in 1:length(I_thal)
        I_init_thal[i] = copy(defs[I_thal[i]]);
    end
    

    #layer 5 stp time scale
    id = findall(x -> occursin("Layer_5", String(Symbol(x))), ps);

    id2 = findall(x -> occursin("exci", String(Symbol(x))), ps);

    id3 = findall(x -> occursin("τ₃", String(Symbol(x))), ps);

    id4 = intersect(id,id2,id3)


    for ii=1:length(id4)
        defs[ps[id4[ii]]]=300
    end

    

    #layer 2 AC stp time scale
    id = findall(x -> occursin("Layer_2_3_ac", String(Symbol(x))), ps);

    id2 = findall(x -> occursin("exci", String(Symbol(x))), ps);

    id3 = findall(x -> occursin("τ₃", String(Symbol(x))), ps);

    id4 = intersect(id,id2,id3)


    for ii=1:length(id4)
        defs[ps[id4[ii]]]=300
    end
    


end


begin
    ps=parameters(sys);
    idp=findall(x -> occursin("pulse", String(Symbol(x))), ps);
    idA=findall(x -> occursin("InfC_A", String(Symbol(x))), ps);
    idB=findall(x -> occursin("InfC_B", String(Symbol(x))), ps);

    idAp = intersect(idp,idA);
    idBp = intersect(idp,idB);
    ppA = ps[idAp[1]];
    ppA_ = ps[idAp[2]];
    ppB = ps[idBp[1]];
    
    #defs = ModelingToolkit.get_defaults(sys);

    #input pulse amp from Inferior colliculus
    defs[ppA] = 20#16#14;
    defs[ppA_] = 20
    defs[ppB] = 65#65#35#16#14;
    
    # background currents for cortical neurons of layer 2-3
    I_bg_amp=0
    id1=findall(x -> occursin("I_bg", String(Symbol(x))), ps);
    id2=findall(x -> occursin("exci", String(Symbol(x))), ps);
    id3=findall(x -> occursin("2_3_ac", String(Symbol(x))), ps);
    id_bg=intersect(id1,id2,id3);
    I=I_bg_amp*rand(length(id_bg));
    
    I_ar = ps[id_bg];

    for i in 1:length(id_bg)
        defs[I_ar[i]] = I[i];
    end
    
    

    # GABAergic synapse timescale.
    propofol_param_ts=1#2.5#2 #2

    for i in 1:length(inh_t)
        defs[inh_t[i]] = init_ts[i]*propofol_param_ts;
    end

    # GABAergic synapse assymptotic current.
    propofol_param_g=1#2.5#2 #3

    for i in 1:length(inh_t)
        defs[inh_g[i]] = init_g[i]*propofol_param_g;
    end

    #Input to Nrt
    id1=findall(x -> occursin("I_bg", String(Symbol(x))), ps);
    id2=findall(x -> occursin("Nrt", String(Symbol(x))), ps);
    id_bg_nrt=intersect(id1,id2);
    I_nrt=ps[id_bg_nrt];
    I_nrt_amp=3 - 3*propofol_param_g     
    
    for i in 1:length(I_nrt)
        defs[I_nrt[i]] = I_nrt_amp;
    end

    # background currents for Thal
    for i in 1:length(I_thal)
        defs[I_thal[i]] = (I_init_thal[i]-0.25)*propofol_param_g + 0.25;
    end

    id = findall(x -> occursin("w_Nrt", String(Symbol(x))), ps)
    for ii = 1:length(id)
     #   defs[ps[id[ii]]]=0
    end

    @time pv=ModelingToolkit.MTKParameters(sys,defs);
    

end

# ╔═╡ bd3dd6d3-e456-4256-a831-a41d0f182123
begin

prob_new=remake(prob; u0 = u0, p = pv, tspan = (0.0, tspan))#2*tspan));#1000));
#prob_new=remake(prob;  p = pv, tspan = (0.0, 2*tspan))
#prob = ODEProblem(sys, [], (0.0, 1700), []);#1700
@time sol = solve(prob_new, Vern7(), saveat=0.1);

	
end




# ╔═╡ 1fc91dfd-91b5-43fd-9b68-6db51f3cf7e4
begin

    fig = Figure();
    ax = Axis(fig[1,1]; xlabel = "time (ms)", ylabel = "Meanfield voltage (mv)")

	exci_nrn=get_exci_neurons(Layer_6_A);
    #exci_nrn=get_neurons(Layer_6_A);
epsc_6 = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_6 = hcat(epsc_6,G)
end

 
exci_nrn=get_exci_neurons(Layer_6_B);
#exci_nrn=get_neurons(Layer_6_B);
epsc_6_ = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_6_ = hcat(epsc_6_,G)
end	

#epsc_5 = state_timeseries(Layer_5_A, sol, "G")

lfp_6 = mean(epsc_6,dims=2)  

exci_nrn=get_exci_neurons(Layer_2_3_A);
#exci_nrn=get_neurons(Layer_2_3_A);
epsc_2 = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_2 = hcat(epsc_2,G)
end
	

exci_nrn=get_exci_neurons(Layer_2_3_B);
#exci_nrn=get_neurons(Layer_2_3_B);
epsc_2_ = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_2_ = hcat(epsc_2_,G)
end	

	
lfp_2 = mean(epsc_2,dims=2)

#exci_nrn=get_exci_neurons(Layer_4_A);
exci_nrn=get_neurons(Layer_4_A);
epsc_4 = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_4 = hcat(epsc_4,G)
end


#exci_nrn=get_exci_neurons(Layer_4_B);
exci_nrn=get_neurons(Layer_4_B);
epsc_4_ = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_4_ = hcat(epsc_4_,G)
end

	

#epsc_5 = state_timeseries(Layer_5_A, sol, "G")

lfp_4 = mean(epsc_4,dims=2)
lfp_4_ = mean(epsc_4_,dims=2)

exci_nrn=get_exci_neurons(Layer_5_A);
#exci_nrn=get_neurons(Layer_2_3_A);
epsc_5 = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_5 = hcat(epsc_5,G)
end

lfp_5 = mean(epsc_5,dims=2)

exci_nrn=get_exci_neurons(Layer_5_B);
#exci_nrn=get_neurons(Layer_2_3_A);
epsc_5_ = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_5_ = hcat(epsc_5_,G)
end

lfp_5 = mean(epsc_5,dims=2)

epsc_5_gstp = state_timeseries(exci_nrn[1],sol,"Gₛₜₚ")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"Gₛₜₚ")
    epsc_5_gstp = hcat(epsc_5_gstp,G)
end



neuron_set = get_neurons(Layer_6_A) ## extract neurons from a composite blocks 
#stackplot(neuron_set[1:25],sol)
#mnv = meanfield_timeseries(Thal_core_A, sol)

lines!(ax,sol.t,lfp_6[:,1])

epsc_t = state_timeseries(Thal_core_A, sol, "G")

lfp_t = mean(epsc_t,dims=2)
#lines!(ax,sol.t,[lfp_5[:,1],lfp_t[:,1]])
lines!(ax,sol.t,lfp_t[:,1])

epsc_t_ = state_timeseries(Thal_core_B, sol, "G")

lfp_t_ = mean(epsc_t_,dims=2)

epsc_nrt = state_timeseries(Nrt_A, sol, "G")

lfp_nrt = mean(epsc_nrt,dims=2)



exci_nrn=get_exci_neurons(Layer_2_3_ac);
#exci_nrn=get_neurons(Layer_2_3_A);
epsc_2_ac = state_timeseries(exci_nrn[1],sol,"G")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"G")
    epsc_2_ac = hcat(epsc_2_ac,G)
end
inh_nrn=get_inh_neurons(Layer_2_3_ac);
#exci_nrn=get_neurons(Layer_2_3_A);
for nn in 2:length(inh_nrn)
    G = state_timeseries(inh_nrn[nn],sol,"G")./30
    epsc_2_ac = hcat(epsc_2_ac,G)
end


lfp_2_ac = mean(epsc_2_ac,dims=2)

exci_nrn=get_exci_neurons(Layer_2_3_ac);
epsc_2_gstp = state_timeseries(exci_nrn[1],sol,"Gₛₜₚ")
for nn in 2:length(exci_nrn)
    G = state_timeseries(exci_nrn[nn],sol,"Gₛₜₚ")
    epsc_2_gstp = hcat(epsc_2_gstp,G)
end
#lines!(ax,sol.t,lfp_nrt[:,1])

#lines!(ax,sol.t,lfp_2[:,1])
#lines!(ax,sol.t,lfp_nrt[:,1])	
#lfp = mean(hcat(epsc_6,epsc_6_,epsc_4,epsc_4_,epsc_2,epsc_2_,epsc_t,epsc_t_),dims=2)	
lfp = mean(hcat(epsc_6,epsc_6_,epsc_4,epsc_4_,epsc_2,epsc_2_,epsc_5,epsc_5_),dims=2)	
#lfp = mean(hcat(epsc_6,epsc_4,epsc_2,epsc_5),dims=2)
lines!(ax,sol.t,lfp[:,1])
fig
end

begin
neuron_set = get_neurons(Layer_5_A) 
stackplot(neuron_set[1:31],sol)

end

begin
neuron_set = get_neurons(Layer_2_3_ac) 
stackplot(neuron_set[1:61],sol)

end









begin

exci_nrn=get_neurons(Layer_2_3_A);
#exci_nrn=get_neurons(Layer_2_3_A);
v2A = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    v2A = hcat(v2A,v)
end

exci_nrn=get_neurons(Layer_2_3_B);
#exci_nrn=get_neurons(Layer_2_3_A);
v2B = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    v2B = hcat(v2B,v)
end

exci_nrn=get_neurons(Layer_5_A);
#exci_nrn=get_neurons(Layer_2_3_A);
v5A = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    v5A = hcat(v5A,v)
end

exci_nrn=get_neurons(Layer_5_B);
#exci_nrn=get_neurons(Layer_2_3_A);
v5B = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    v5B = hcat(v5B,v)
end

exci_nrn=get_neurons(Thal_mat_A);
#exci_nrn=get_neurons(Layer_2_3_A);
vtA = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    vtA = hcat(vtA,v)
end

exci_nrn=get_neurons(Thal_mat_B);
#exci_nrn=get_neurons(Layer_2_3_A);
vtB = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    vtB = hcat(vtB,v)
end

exci_nrn=get_neurons(Layer_2_3_ac);
#exci_nrn=get_neurons(Layer_2_3_A);
v2ac = state_timeseries(exci_nrn[1],sol,"V")
for nn in 2:length(exci_nrn)
    v = state_timeseries(exci_nrn[nn],sol,"V")
    v2ac = hcat(v2ac,v)
end
end

begin
open("V5Bawake.txt", "w") do io
    writedlm(io, v5B[tin[1]:tfin[1],:], ",")
end
end


 begin
open("lfp_local_oddball_propofol_core.txt", "w") do io
    writedlm(io, lfp, ",")
end
end

begin
open("lfp_local_oddball_propofol_matrix_anim.txt", "w") do io
    writedlm(io, lfp_2_ac[tin[1]:tfin[1],:], ",")
end
end

begin
    open("lfp_global_time.txt", "w") do io
        writedlm(io, sol.t, ",")
    end
    end


# ╔═╡ Cell order:
# ╠═170d98f5-1177-4d97-9cb3-c899c5aa5f41
# ╠═0d29fc66-498d-4637-a3e2-ed16ab1d8bae
# ╠═2ea6657c-0336-11f0-256b-0d2ed24ac5e6
# ╠═7c04d938-c736-428c-923f-fc1e4c9a46bb
# ╠═90f96021-80e1-4cd8-b5e7-9818c2b6c21f
# ╠═bd3dd6d3-e456-4256-a831-a41d0f182123
# ╠═1fc91dfd-91b5-43fd-9b68-6db51f3cf7e4
   