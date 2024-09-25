using Neuroblox, DifferentialEquations, Distributions, Graphs, MetaGraphs, Random

# First focus is on producing panels from Figure 1 of the PING network paper.

# Setup parameters from the supplemental material
μ_E = 1.5
σ_E = 0.15
μ_I = 0.8
σ_I = 0.08

# Define the PING network neuron numbers
NE_driven = 2
NE_other = 14
NI_driven = 4
N_total = NE_driven + NE_other + NI_driven

# First, create the 20 driven excitatory neurons
exci_driven = [PINGNeuronExci(name=Symbol("ED$i"), I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NE_driven]
exci_other  = [PINGNeuronExci(name=Symbol("EO$i")) for i in 1:NE_other]
inhib       = [PINGNeuronInhib(name=Symbol("ID$i"), I_ext=rand(Normal(μ_I, σ_I))) for i in 1:NI_driven]

# Create the network
g = MetaDiGraph()
add_blox!.(Ref(g), vcat(exci_driven, exci_other, inhib))

# Extra parameters
p = @parameters N=N_total g_II=0.2 g_IE=0.6 g_EI=0.6

for i = 1:NE_driven+NE_other
    for j = NE_driven+NE_other+1:N_total
        add_edge!(g, i, j, Dict(:weight => g_EI/N))
        add_edge!(g, j, i, Dict(:weight => g_IE/N))
    end
end

for i = NE_driven+NE_other+1:N_total
    for j = NE_driven+NE_other+1:N_total
        add_edge!(g, i, j, Dict(:weight => g_II/N))
    end
end

@named sys = system_from_graph(g, p)
sys = structural_simplify(sys)
prob = ODEProblem(sys, [], (0.0, 20.0))
@time sol = solve(prob, Tsit5(), saveat=0.1)

using Peaks, Plots

exci_voltages = reduce(hcat, ModelingToolkit.getu(sol, vcat([Symbol("ED$i"*"₊V") for i in 1:NE_driven], [Symbol("EO$i"*"₊V") for i in 1:NE_other]))(sol))'
inhib_voltages = reduce(hcat, ModelingToolkit.getu(sol, [Symbol("ID$i"*"₊V") for i in 1:NI_driven])(sol))'

plot(exci_voltages)
plot(inhib_voltages)

maxima(inhib_voltages[:, 1])