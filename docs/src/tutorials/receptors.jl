# # Receptor Dynamics and Interventions

# ## Introduction
# Between any pre- and postsynaptic neurons in a Neuroblox model, there exist a synapse that captures the dynamics of the transmitter release and the receptor on the postsynaptic side.
# Here we will see how to choose synapses when connecting neurons and how to change synapse parameters to model interventions like drugs that alter receptor dynamics.

# ## Receptors between two neurons
# Firstly we make a reciprocal connection between an excitatory and inhibitory neuron. The excitatory projections contains both AMPA and NMDA receptors, whereas the inhibitory one is mediated by a GABA A receptor.

using Neuroblox
using OrdinaryDiffEqVerner
using CairoMakie

@graph g begin
    @nodes begin
        ne = HHNeuronExci(; I_bg=8)
        ni = HHNeuronInhib(; I_bg=0.4)
    end
    @connections begin
        ne => ni , [weight=0.1, synapse=Glu_AMPA_Synapse(name=:ampa)]
        ne => ni , [weight=0.1, synapse=NMDA_Synapse(name=:nmda)]
        ni => ne , [weight=1, synapse=GABA_A_Synapse(name=:gabaa)]
     end
end

# Using the `stackplot` recipe we plot the voltage of both neurons (blue for excitatory and red for inhibitory).

prob = ODEProblem(g, [], (0, 1000))
sol = solve(prob, Vern7())

stackplot([ne, ni], sol)

# Now let us change the inhibitory connection to a GABA B receptor. The excitatory connection remains the same.

@graph g begin
    @nodes begin
        ne = HHNeuronExci(; I_bg=8)
        ni = HHNeuronInhib(; I_bg=0.4)
    end
    @connections begin
        ne => ni , [weight=0.1, synapse=Glu_AMPA_Synapse(name=:glu)]
        ne => ni , [weight=0.1, synapse=NMDA_Synapse(name=:nmda)]
        ni => ne , [weight=1, synapse=GABA_B_Synapse(name=:gabab)]
    end
end
prob = ODEProblem(g, [], (0, 1000))
sol = solve(prob, Vern7())

stackplot([ne, ni], sol)

# Notice the changes in the firing rates of both neurons between the GABA A and GABA B cases.
# The GABA B receptor operates on a longer timescale for both during activation and deactivation. It takes longer to activate but also exhibits slower closing times compared to GABA A. Therefore the inhibition on the excitatory neuron remains active for longer, thus supressing its firing which in turn suppresses the excitation on the inhibitory neuron.

# ## Changing E-I balance in a cortical microcircuit
# Besides choosing the receptor types in neuronal connections, we can also change receptor parameters in circuit models, to simulate the effect of interventions, most commonly drugs.
# Here we will build a model containing a cortical microcircuit, derived from layers 2-3 of posterior cortex. We will add an ascending input to this microcircuit coming from the brainstem and working as a pacemaker.

@graph g begin
    @nodes begin
        Brainstem = NGNMM_theta(Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26)
        Layer_2_3 = Cortical(N_wta=20, N_exci=5, density=0.05, weight=1, I_bg_ar=4, G_syn_inhib=4, τ_inhib=70, G_syn_ff_inhib=1.5)
    end
    @connections begin
        Brainstem => Layer_2_3, [weight = 20]
    end
end

prob = ODEProblem(g, [], (0.0, 1000.0))
sol = solve(prob, Vern7());

prob_EI = @experiment prob begin
    @setup begin
        p_I = 1.5 ## GABA factor
        p_E = 3.0 ## AMPA factor
    end
    GABA_A_Synapse, τ₂ -> τ₂ * p_I
    GABA_A_Synapse, G_syn -> G_syn * p_I

    Glu_AMPA_Synapse, τ₂ -> τ₂ * p_E
    Glu_AMPA_Synapse, G_syn -> G_syn * p_E
end

# After solving the circuit model in its control condition, we design an `@experiment` above to simulate an intervention on both the GABA A and the AMPA receptors, which are the default receptos in the `Cortical` blox. This way we change the E-I balance of the circuit by affecting both components.

sol_EI = solve(prob_EI, Vern7());

fig = Figure()
ax = Axis(fig[1,1], title="Cortical LFP")
meanfield!(ax, Layer_2_3, sol; label="Control")
meanfield!(ax, Layer_2_3, sol_EI; color=:red, label="E-I Intervention")
axislegend(position=:rt, framevisible = false)
fig

# We solved the experiment problem and then used the solution objects to plot the cortical LFP activity for both conditions.