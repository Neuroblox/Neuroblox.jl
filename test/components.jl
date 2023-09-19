using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random

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

first create a single canonical micro circuit and simulate. Then create a two region model and connect two
to form the circuit that is given in Figure 4 of Bastos et al. 2015.
"""

# canonical micro circuit based on single Jansen-Rit blox
@named singleregion = CanonicalMicroCircuitBlox()
eqs = []
for bi in singleregion.bloxinput
    push!(eqs, bi ~ 0)
end
@named singleregionODE = ODESystem(eqs, systems=[singleregion.odesystem])
singleregionODE = structural_simplify(singleregionODE)
sol = simulate(singleregionODE, [], (0.0, 10.0), [])
# TODO: it would be nicer if this was without the singleregion namespace...
@test sol[!,"singleregion₊dp₊x(t)"][end] + sol[!,"singleregion₊ii₊x(t)"][end] ≈ -5.159425345927338


# connect multiple canonical micro circuits according to Figure 4 in Bastos et al. 2015
@named r1 = CanonicalMicroCircuitBlox()
@named r2 = CanonicalMicroCircuitBlox()
@named jr = JansenRitCBlox()

g = MetaDiGraph()
add_vertex!(g, Dict(:blox => r1)) # V1 (see fig. 4 in Bastos et al. 2015)
add_vertex!(g, Dict(:blox => r2)) # V4 (see fig. 4 in Bastos et al. 2015)
add_edge!(g, 1, 2, :weightmatrix, [0 1 0 0; # superficial pyramidal to spiny stellate
                                   0 0 0 0;
                                   0 0 0 0;
                                   0 1 0 0]) # superficial pyramidal to deep pyramidal
# define connections from column (source) to row (sink)
add_edge!(g, 2, 1, :weightmatrix, [0 0 0  0; 
                                   0 0 0 -1;
                                   0 0 0 -1;
                                   0 0 0  0])

@named cmc_network = ODEfromGraph(g)
cmc_network = structural_simplify(cmc_network)

sol = simulate(cmc_network, [], (0.0, 10.0), [])
@test sum(sol[end, 2:end]) ≈ -4827.086868187682

# now add a Neural mass model with one output and one input
add_vertex!(g, Dict(:blox => jr))
add_edge!(g, 3, 1, :weightmatrix, [0; 0; 0; 1])
add_edge!(g, 1, 3, :weightmatrix, [[1 0 0 0];])
add_edge!(g, 3, 3, :weight, -1)

@named cmc_network2 = ODEfromGraph(g)
cmc_network2 = structural_simplify(cmc_network2)
sol = simulate(cmc_network2, [], (0.0, 10.0), [])
@test sum(sol[end, 2:end]) ≈ -4823.399802568824

# now connect canonical micro circuits with symbolic weight matrices
g = MetaDiGraph()
add_vertex!(g, Dict(:blox => r1))
add_vertex!(g, Dict(:blox => r2))

A_forward = [0 1 0 0;
             0 0 0 0;
             0 0 0 0;
             0 1 0 0]
A_backward = [0 0 0  0;
              0 0 0 -1;
              0 0 0 -1;
              0 0 0  0]
@parameters wm_forward[1:length(A_forward)] = vec(A_forward) 
add_edge!(g, 1, 2, :weightmatrix, reshape(wm_forward, 4, 4))
@parameters wm_backward[1:length(A_backward)] = vec(A_backward) 
add_edge!(g, 2, 1, :weightmatrix, reshape(wm_backward, 4, 4))

@named cmc_network = ODEfromGraph(g)
cmc_network = structural_simplify(cmc_network)

"""
Components Test for Cortical-Subcortical Jansen-Rit blox
    Cortical: PFC (Just Pyramidal Cells (PY), no Exc. Interneurons or Inh. Interneurons)
    Subcortical: Basal Ganglia (GPe, STN, GPi) + Thalamus
"""

# Create Regions
@named GPe       = JansenRitCBlox(τ=0.04, H=20, λ=400, r=0.1)
@named STN       = JansenRitCBlox(τ=0.01, H=20, λ=500, r=0.1)
@named GPi       = JansenRitCBlox(τ=0.014, H=20, λ=400, r=0.1)
@named Thalamus  = JansenRitSCBlox(τ=0.002, H=10, λ=20, r=5)
@named PFC       = JansenRitSCBlox(τ=0.001, H=20, λ=5, r=0.15)

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
	 nn = QIFNeuronBlox(name=Symbol("nrn$ii"),C=30.0,E_syn=-10,G_syn=1,ω=rand(Cauchy(ω₀,Δω)),τ=35)
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
# N = 6   # 6 neurons

# # neuron properties
# I_in = ones(N);             # same input current to all
# τ = 5*collect(1:N);         # increasing membrane time constant
# # synaptic properties 
# syn_amp = 0.4*ones(N, N); # synaptic amplitudes
# syn_τ = 5*ones(N)
# nrn_network=[]
# nrn_spiketimes=[]

# for i = 1:N
#     nn = LIFNeuronBlox(name=Symbol("lif$i"), I_in=I_in[i], τ=τ[i])
#     push!(nrn_network, nn.odesystem)
# 	push!(nrn_spiketimes, nn.odesystem.st)
# end

# # connect the neurons
# @named syn_net = spikeconnections(sys=nrn_network, psp_amplitude=syn_amp, τ=syn_τ, spiketimes=nrn_spiketimes)

# sim_dur =  50.0
# prob = ODEProblem(structural_simplify(syn_net), [], (0.0, sim_dur), [])
# sol = solve(prob, AutoVern7(Rodas4())) #pass keyword arguments to solver

# @test length(sol.prob.p[end]) == 5
# @test length(sol.prob.p[21]) == 11



"""
van_der_pol.jl test

Test for van der Pol generator.
"""

@named VdP = van_der_pol()

prob_vdp = SDEProblem(VdP,[0.1,0.1],[0.0, 20.0],[])
sol = solve(prob_vdp,EM(),dt=0.1)
@test sol.retcode == SciMLBase.ReturnCode.Success

"""
stochastic.jl test

Test for OUBlox generator.
"""

@named ou1 = OUBlox()
sys = [ou1.odesystem]
eqs = [sys[1].jcn ~ 0.0]
@named ou1connected = compose(System(eqs;name=:connected),sys)
ousimpl = structural_simplify(ou1connected)
prob_ou = SDEProblem(ousimpl,[],(0.0,10.0))
sol = solve(prob_ou,alg_hints = [:stiff])
@test sol.retcode == SciMLBase.ReturnCode.Success
@test std(sol[1,:]) > 0.0 # there should be variance

# connect OU process to Neural Mass Blox
@named jr = JansenRitCBlox()
sys = [ou1.odesystem, jr.odesystem]
eqs = [sys[1].jcn ~ 0.0, sys[2].jcn ~ sys[1].x]
@named ou1connected = compose(System(eqs;name=:connected),sys)
ousimpl = structural_simplify(ou1connected)
prob_oujr = SDEProblem(ousimpl,[],(0.0,10.0))
sol = solve(prob_oujr, alg_hints = [:stiff])
@test sol.retcode == SciMLBase.ReturnCode.Success
@test std(sol[2,:]) > 0.0 # there should be variance

# test OU coupling blox
@named oucp = OUCouplingBlox(μ=2.0, σ=1.0, τ=1.0)
sys = [ou1.odesystem, oucp.odesystem]
eqs = [sys[1].jcn ~ 0.0, sys[2].jcn ~ sys[1].x]
@named ou1connected = compose(System(eqs;name=:connected),sys)
ousimpl = structural_simplify(ou1connected)
prob_oucp = SDEProblem(ousimpl,[],(0.0,10.0))
sol = solve(prob_oucp, alg_hints = [:stiff])
@test sol.retcode == SciMLBase.ReturnCode.Success
@test std(sol[1,:].*sol[2,:]) > 0.0 # there should be variance

# test a larger system
@named ou1 = OUBlox(μ=0.0, σ=1.0, τ=3.0)
@named ou2 = OUBlox(μ=0.0, σ=1.0, τ=3.0)
@named oucp1 = OUCouplingBlox(μ=-0.1, σ=0.02, τ=10)
@named oucp2 = OUCouplingBlox(μ=-0.2, σ=0.02, τ=10)
sys = [ou1.odesystem, ou2.odesystem, oucp1.odesystem, oucp2.odesystem]
eqs = [sys[1].jcn ~ oucp1.connector,
        sys[2].jcn ~ oucp2.connector,
        sys[3].jcn ~ ou2.connector,
        sys[4].jcn ~ ou1.connector]
@named ouconnected = compose(System(eqs;name=:connected),sys)
ousimpl = structural_simplify(ouconnected)
prob_ouconnect = SDEProblem(ousimpl,[0,0,-0.1,-0.2],(0.0,100.0))
sol = solve(prob_ouconnect, alg_hints = [:stiff])
@test sol.retcode == SciMLBase.ReturnCode.Success
@test std(sol[1,:].*sol[2,:]) > 0.0 # there should be variance
@test cor(sol[1,:],sol[2,:]) < 0.0 # Pearson correlation should be negative

"""
wilson_cowan test

Test for Wilson-Cowan model
"""
@named WC = WilsonCowanBlox()
sys = [WC.odesystem]
eqs = [sys[1].jcn ~ 0.0, sys[1].P ~ 0.0]
@named WC_sys = ODESystem(eqs,systems=sys)
WC_sys_s = structural_simplify(WC_sys)
prob = ODEProblem(WC_sys_s, [], (0,sim_dur), [])
sol = solve(prob,AutoVern7(Rodas4()),saveat=0.01)
#@test sol[1,end] ≈ 0.17513685727060388

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

@test sol[1,10] ≈ -0.6246710908910991

"""
CorticalBlox test
"""
@named cb = CorticalBlox(N_wta=6, N_exci=5)
cb_simpl = structural_simplify(cb.odesystem)
@test length(states(cb_simpl)) == 216
prob = ODEProblem(cb_simpl, [], (0, 20))
sol = solve(prob, Vern7(), saveat=0.5)
@test size(sol) == (216, 41)

global_ns = :g # global namespace
@named cb = CorticalBlox(N_wta=6, N_exci=5; namespace=global_ns)
fn = "../examples/image_data.csv"
@named stim = ImageStimulus(fn; namespace=global_ns, t_stimulus=1, t_pause=0.5)
g = MetaDiGraph()
add_blox!(g, stim)
add_blox!(g, cb)
add_edge!(g, 1, 2, :weight, 1)
sys = system_from_graph(g; name=global_ns)
sys_simpl = structural_simplify(sys)
prob = ODEProblem(sys_simpl, [], (0, 10); tofloat=false)
sol = solve(prob, Vern7())
@test sol isa Any

"""
SuperCortical
"""
@named sc  = SuperCortical(; N_cb=2, N_wta=6, out_degree=3)
sc_simpl = structural_simplify(sc.odesystem)
@test length(states(sc_simpl)) == 432
prob = ODEProblem(sc_simpl, [], (0, 20))
sol = solve(prob, Vern7(), saveat=0.5)
@test size(sol) == (432, 41)

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


"""
test for HHNeuronExciBlox, HHNeuronInhibBlox and SynapticConnections
"""
nn1 = HHNeuronExciBlox(name=Symbol("nrn1"), I_in=3, freq=4)
nn2 = HHNeuronExciBlox(name=Symbol("nrn2"), I_in=2, freq=6)
nn3 = HHNeuronInhibBlox(name=Symbol("nrn3"), I_in=2, freq=3)
assembly = [nn1, nn2, nn3]

# Adjacency matrix : 
#adj = [0 1 0
#       0 0 1
#       0.2 0 0]
g = MetaDiGraph()
add_blox!.(Ref(g), assembly)
add_edge!(g, 1, 2, :weight, 1)
add_edge!(g, 2, 3, :weight, 1)
add_edge!(g, 3, 1, :weight, 0.2)
      
@named neuron_net = system_from_graph(g)
sol = simulate(structural_simplify(neuron_net), [], (0, 10), [], Vern7())

@test neuron_net isa ODESystem
@test sol[:,1][end] ≈ 10.0

"""
test for WinnerTakeAllBlox
"""
inp = 5*rand(5)
@named wta= WinnerTakeAllBlox(I_in=inp)
sys = wta.odesystem
wta_simp=structural_simplify(sys)
prob = ODEProblem(wta_simp,[],(0,10))
sol = solve(prob, Vern7(), saveat=0.1)

@test typeof(wta_simp)==ODESystem
@test sol.t[end]==10