using Neuroblox, OrdinaryDiffEq, StochasticDiffEq, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, Random


"""
neuralmass.jl test
"""

# Create Regions
@named Str = jansen_ritC(τ=0.0022, H=20, λ=300, r=0.3)
@named GPe = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
@named STN = jansen_ritC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi = jansen_ritSC(τ=0.014, H=20, λ=400, r=0.1)
@named Th  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
@named EI  = jansen_ritSC(τ=0.01, H=20, λ=5, r=5)
@named PY  = jansen_ritSC(τ=0.001, H=20, λ=5, r=0.15)
@named II  = jansen_ritSC(τ=2.0, H=60, λ=5, r=5)

# Connect Regions through Adjacency Matrix
blox = [Str, GPe, STN, GPi, Th, EI, PY, II]
sys = [s.odesystem for s in blox]
connect = [s.connector for s in blox]

@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 0 0 0 0 0 0 0;
            -0.5*C_BG_Th -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 -0.5*C_BG_Th 0 0 0 0 C_Cor_BG_Th 0;
            0 -0.5*C_BG_Th C_BG_Th 0 0 0 0 0;
            0 0 0 -0.5*C_BG_Th 0 0 0 0;
            0 0 0 0 C_BG_Th_Cor 0 6*C_Cor 0;
            0 0 0 0 0 4.8*C_Cor 0 -1.5*C_Cor;
            0 0 0 0 0 0 1.5*C_Cor 3.3*C_Cor]

@named CBGTC_Circuit_lin = LinearConnections(sys=sys, adj_matrix=adj_matrix_lin, connector=connect)

sim_dur = 10.0 # Simulate for 10 Seconds
mysys = structural_simplify(CBGTC_Circuit_lin)
sol = simulate(mysys, [], (0.0, sim_dur), [])
@test sol[!, "GPi₊x(t)"][4] ≈ 0.976257006970988

"""
testing random inital conditions for neural mass blox
"""

sol = simulate(mysys, random_initials(mysys,blox),(0.0, sim_dur), [])
@test size(sol)[2] == 17 # make sure that all the states are simulated (16 + timestamp)

"""
Canonical micro circuit tests 
"""

@named region = cmc_singleregion()
region = structural_simplify(region.odesystem)
sol = simulate(region, [], (0.0, 10.0), [])
@test sol[!,"x3(t)"][end] + sol[!,"x4(t)"][end] ≈ -5.159425345927339

# canonical micro circuit based on single Jansen-Rit blox
@named singleregion = cmc()
singleregion = structural_simplify(singleregion.odesystem)
sol = simulate(singleregion, [], (0.0, 10.0), [])
@test sol[!,"singleregion_dp₊x(t)"][end] + sol[!,"singleregion_ii₊x(t)"][end] ≈ -5.159425345927338

# connect multiple canonical micro circuits
@named r1 = cmc()
@named r2 = cmc()
@named r3 = cmc()

regions = [r1, r2, r3]
nr = length(regions)
A = Array{Matrix{Float64}}(undef, nr, nr);
Random.seed!(1234)
for i = 1:nr
    for j = 1:nr
        if i == j continue end
        nodes_source = nv(regions[i].lngraph.graph)
        nodes_sink = nv(regions[j].lngraph.graph)
        A[i, j] = rand(nodes_source, nodes_sink)
    end
end
@named manyregions = connectcomplexblox(regions, A)
manyregions = structural_simplify(manyregions)
sol = simulate(manyregions, [], (0.0, 10.0), [])
@test_broken sol[!,"r1_ss₊x(t)"][10] + sol[!,"r2_sp₊y(t)"][10] + sol[!,"r3_dp₊x(t)"][10] ≈ -350.89065248035655


"""
Components Test for Cortical-Subcortical Jansen-Rit blox
    Cortical: PFC (Just Pyramidal Cells (PY), no Exc. Interneurons or Inh. Interneurons)
    Subcortical: Basal Ganglia (GPe, STN, GPi) + Thalamus
"""

# Create Regions
@named GPe       = jansen_ritC(τ=0.04, H=20, λ=400, r=0.1)
@named STN       = jansen_ritC(τ=0.01, H=20, λ=500, r=0.1)
@named GPi       = jansen_ritC(τ=0.014, H=20, λ=400, r=0.1)
@named Thalamus  = jansen_ritSC(τ=0.002, H=10, λ=20, r=5)
@named PFC       = jansen_ritSC(τ=0.001, H=20, λ=5, r=0.15)

# Connect Regions through Adjacency Matrix
blox = [GPe, STN, GPi, Thalamus, PFC]
sys = [s.odesystem for s in blox]
connect = [s.connector for s in blox]

@parameters C_Cor=60 C_BG_Th=60 C_Cor_BG_Th=5 C_BG_Th_Cor=5

adj_matrix_lin = [0 C_BG_Th 0 0 0;
            -0.5*C_BG_Th 0 0 0 C_Cor_BG_Th;
            -0.5*C_BG_Th C_BG_Th 0 0 0;
            0 0 -0.5*C_BG_Th 0 0;
            0 0 0 C_BG_Th_Cor 0]

@named CBGTC_Circuit_lin = LinearConnections(sys=sys, adj_matrix=adj_matrix_lin, connector=connect)
sim_dur = 10.0 # Simulate for 10 Seconds
mysys = structural_simplify(CBGTC_Circuit_lin)
sol = simulate(mysys, [], (0.0, sim_dur), [])


"""
complex neural mass model test (next generation neural mass model)
This test generates a neural mass model using the kuramoto order parameter
to capture within-population synchrony. A model is generated and then
the phase of oscillations is computed (ψ) along with synchrony (R). 
This model has no input, and therefore oscillations and synchrony should
tend toward zero.
"""
@named macroscopic_model = next_generation(C=30, Δ=1.0, η_0=5.0, v_syn=-10, alpha_inv=35, k=0.105)
sim_dur = 1000.0 
sol = simulate(structural_simplify(macroscopic_model.odesystem), [0.5 + 0.0im, 1.6 + 0.0im], (0.0, sim_dur), [], Tsit5(); saveat=0.01,reltol=1e-4,abstol=1e-4)

C=30
W = (1 .- conj.(sol[!,"Z(t)"]))./(1 .+ conj.(sol[!,"Z(t)"]))
R = (1/(C*pi))*(W+conj.(W))/2
ψ = log.(sol[!,"Z(t)"]./R)/im

@test norm.(R[length(R)]) < 0.1


"""
thetaneuron.jl test

Test approach: generate a network of theta neurons and connect them through an all-to-all
adjacency matrix via a spike function. Then compute the real part of the Kuramoto order parameter.
The average of this parameter should be close to zero as synchrony varies in the network from a positive
to a negative amplitude. 
"""

# Generate Theta Network
network = [] 
N = 50
for i = 1:N
    η  = rand(Cauchy(1.0, 0.05)) # Constant Drive
    @named neuron = Neuroblox.theta_neuron(name=Symbol("neuron$i"), η=η, α_inv=1.0, k=-2.0)
    push!(network, neuron)
end

# Create Circuit
adj_matrix = (1/N)*ones(N,N)
n = 3
a_n = 2.0^n*(factorial(n)^2.0)/(factorial(2*n))
@named theta_circuit = LinearConnections(sys=network, adj_matrix=adj_matrix, connector=[a_n*(1-cos(neuron.θ))^n for neuron in network])

sim_dur = 10.0 # Simulate for 10 Seconds
sol = Neuroblox.simulate(structural_simplify(theta_circuit), [], (0.0, sim_dur), [])
R = real(exp.(im*sol[!, "neuron1₊θ(t)"]))

@test abs(Statistics.mean(R)) < 0.6

"""
qif_neuron.jl and synaptic_network.jl test

This test generates a network of quadratic integrate and fire neurons 
using qif_neuron.jl and connects them with synapses using synaptic_network.jl
This should successfully generate a structurally simplified ODESystem for 
the entire network. If N is number of neurons and S+1 is the number of state variables
for each neuron (S internal variables and 1 synaptic input), then the total number of states
for the resulting ODESystem for network should be N*S.
"""

#Generate qif neurons
N_nrn = 10	
nrn_network=[]

ω₀ = 0.269
Δω = 0.042

for ii = 1:N_nrn
	 nn = qif_neuron(name=Symbol("nrn$ii"),C=30.0,E_syn=-10,G_syn=1,ω=rand(Cauchy(ω₀,Δω)),τ=35)
     push!(nrn_network,nn.odesystem)
end

# create synaptic network
k = 0.105 #synaptic weight
adj = ones(N_nrn,N_nrn)
for ii = 1:N_nrn
	adj[ii,ii]=0
end
syn = adj.*k/N_nrn

@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn)

@test typeof(syn_net) == ODESystem
@test length(states(syn_net)) == 3*N_nrn

#simulate for 100 s
sim_dur =  100.0
prob = ODEProblem(syn_net, [], (0.0, sim_dur), [])
sol = solve(prob,Rodas5(),saveat=0.01,reltol=1e-4,abstol=1e-4)
@test sol.t[end] == sim_dur

"""
network of LIFs test
"""
N = 6   # 6 neurons

# neuron properties
I_in = ones(N);             # same input current to all
τ = 5*collect(1:N);         # increasing membrane time constant
# synaptic properties 
syn_amp = 0.4*ones(N, N); # synaptic amplitudes
syn_τ = 5*ones(N)
nrn_network=[]
nrn_spiketimes=[]

for i = 1:N
    nn = LIFneuron(name=Symbol("lif$i"), I_in=I_in[i], τ=τ[i])
    push!(nrn_network, nn.odesystem)
	push!(nrn_spiketimes, nn.odesystem.st)
end

# connect the neurons
@named syn_net = spikeconnections(sys=nrn_network, psp_amplitude=syn_amp, τ=syn_τ, spiketimes=nrn_spiketimes)

sim_dur =  50.0
prob = ODEProblem(structural_simplify(syn_net), [], (0.0, sim_dur), [])
sol = solve(prob, AutoVern7(Rodas4())) #pass keyword arguments to solver

@test length(sol.prob.p[end]) == 5
@test length(sol.prob.p[21]) == 11


"""
leaky integrate integrate and fire neuron if_neuron() test.
This test generates 5 excitatory and 1 inhibitory neurons and connects them into winner take all
network. It then simulates their activity.
"""

# generate if_neurons 
# parameters 
Nrns = 6
E_syn=zeros(1,Nrns);	 #synaptic reversal potentials is property of presynaptic neuron
E_syn[6] =-70;

G_syn = 0.4*ones(1,Nrns); #synaptic conductance is property of presynaptic neuron
G_syn[6] = 5;

I_in = zeros(Nrns); #input currents
for ii = 1:5
    I_in[ii] = 25*rand();
end
I_in[6] = 0.85;
freq = zeros(Nrns);
freq[6] = 5;

phase = zeros(Nrns);
phase[6] = pi
    
τ = 5*ones(Nrns); # postsynaptic potential time constants
τ[6] = 70;

nrn_network=[]
for ii = 1:Nrns
    nn = if_neuron(name=Symbol("nrn$ii"),E_syn=E_syn[ii],G_syn=G_syn[ii],I_in=I_in[ii],freq=freq[ii], phase=phase[ii], τ=τ[ii])
    push!(nrn_network,nn.odesystem)
end

# adjacency matrix 
syn =  zeros(Nrns,Nrns);
syn[end,1:end-1].=1;
syn[1:end-1,end].=1;

#connect the neurons
@named syn_net = synaptic_network(sys=nrn_network,adj_matrix=syn)
@test typeof(syn_net) == ODESystem
@test length(states(syn_net)) == 5*Nrns

# simulate
sim_dur =  50.0
sol = simulate(syn_net, [], (0.0, sim_dur), [])
@test sol[end,1] == sim_dur

"""
hh_neuron_excitatory() and hh_neuron_inhibitory() test.
This test generates 5 excitatory and 1 inhibitory hh neurons and connects them into winner take all
network. It then simulates their activity.
"""
hh_nrn_network=[]
for ii = 1:5
    nn = hh_neuron_excitatory(name=Symbol("hh_nrn$ii"),E_syn=0,G_syn=3,I_in=I_in[ii],freq=0, phase=0, τ=5)
    push!(hh_nrn_network,nn)
end
nn = hh_neuron_inhibitory(name=Symbol("hh_nrn6"),E_syn=-70,G_syn=23,I_in=0.85,freq=4, phase=pi, τ=70)
push!(hh_nrn_network,nn)

# adjacency matrix 
syn =  zeros(Nrns,Nrns);
syn[end,1:end-1].=1;
syn[1:end-1,end].=1;

#connect the neurons
@named hh_syn_net = synaptic_network(sys=hh_nrn_network,adj_matrix=syn)
@test typeof(hh_syn_net) == ODESystem

# simulate
sim_dur =  50.0
sol = simulate(hh_syn_net, [], (0.0, sim_dur), [])
@test sol[end,1] == sim_dur

"""
van_der_pol.jl test

Test for van der Pol generator.
"""

@named VdP = van_der_pol()

prob_vdp = SDEProblem(VdP,[0.1,0.1],[0.0, 20.0],[])
sol = solve(prob_vdp,EM(),dt=0.1)
@test sol.retcode == :Success

"""
wilson_cowan test

Test for Wilson-Cowan model
"""
@named WC = wilson_cowan()
sys = [WC.odesystem]
eqs = [sys[1].jcn ~ 0.0, sys[1].P ~ 0.0]
@named WC_sys = ODESystem(eqs,systems=sys)
WC_sys_s = structural_simplify(WC_sys)
prob = ODEProblem(WC_sys_s, [], (0,sim_dur), [])
sol = solve(prob,AutoVern7(Rodas4()),saveat=0.01)
@test sol[1,end] ≈ 0.17513685727060388

"""
Larter-Breakspear model test
"""
@named lb = LarterBreakspearBlox()
sys = [lb.odesystem]
eqs = [sys[1].jcn ~ 0]
@named lb_connect = ODESystem(eqs,systems=sys)
lb_simpl = structural_simplify(lb_connect)

@test length(states(lb_simpl)) == 3

prob = ODEProblem(lb_simpl,[0.5,0.5,0.5],(0,10.0),[])
sol = solve(prob,Tsit5())

@test sol[1,10] ≈ -0.6246712806761001

"""
CorticalBlox test
"""
@named cb = CorticalBlox(nblocks=6,blocksize=6)
@test length(states(cb.odesystem)) == 222
prob = ODEProblem(cb.odesystem, [], (0, 20))
sol = solve(prob, Vern7(), saveat=0.5)
@test size(sol) == (222,41)

"""
ts_outputs.jl test

Test for time-series output tests.
"""
phase_int = phase_inter(0:3,[0.0,1.0,2.0,1.0])
phase_cos_out(ω,t) = phase_cos_blox(ω,t,phase_int)
phase_sin_out(ω,t) = phase_sin_blox(ω,t,phase_int)
@test phase_cos_out(0.1,2.5)≈0.9689124217106447
@test phase_sin_out(0.1,2.5)≈0.24740395925452294

# now test how to connect this time series to a neural mass blox
@named Str2 = jansen_ritC(τ=0.0022, H=20, λ=300, r=0.3)
@parameters phase_input = 0

sys = [Str2.odesystem]
eqs = [sys[1].jcn ~ phase_input]
@named phase_system = ODESystem(eqs,systems=sys)
phase_system_simpl = structural_simplify(phase_system)
phase_ode = ODEProblem(phase_system_simpl,[],(0,3.0),[])

# create callback functions
# we always want to update phase_input to be our phase_cos_out(t)
condition = function (u,t,integrator)
    true
end

function affect!(integrator)
    integrator.p[1] = phase_cos_out(10*pi,integrator.t)
end

cb = DiscreteCallback(condition,affect!)

sol = solve(phase_ode,Tsit5(),callback=cb)
@test sol[2,:][5] ≈ 13.49728948607267
