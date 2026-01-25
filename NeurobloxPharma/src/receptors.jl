"""
    MoradiFullNMDAR(; E=-0.7, k=0.007, V_0=-100, 
                    τ_A=1.69, τ_B0=3.97, τ_C0=41.62, τ_g=1, 
                    w_B=0.65, λ_B=0.0243, λ_C=0.01, a_B=0.7, a_C=34.69, 
                    Mg_O=1, IC_50=4.1, T=37, F=96485.332, R=8.314, z=2, δ=0.8, 
                    Q_A_10=2.2, T_A_0=31.5, Q_B_10=3.68, T_B_0=35, Q_C_10=2.65, 
                    T_C_0=35, Q_g_10=1.52, T_g_0=26, w_C=1 - w_B)

An NMDA receptor model based on [1]. This is the complete model from the paper including voltage-dependent time constants and temperature correction terms in the activation/deactivation dynamics.

```math
\\frac{dA}{dt} = \\text{spk_coeff} z - \\frac{A}{\\text{temp_modifier}(T, Q_A_10, T_A_0) \\tau_A} \\\\
\\frac{dB}{dt} = \\text{spk_coeff} z - \\frac{B}{\\text{temp_modifier}(T, Q_B_10, T_B_0) \\text{time_constant_B}(V_{post})} \\\\
\\frac{dC}{dt} = - \\frac{C}{\\text{temp_modifier}(T, Q_C_10, T_C_0) \\text{time_constant_C}(V_{post})} \\\\
\\frac{dg}{dt} = (w_C C + w_B B) \\frac{\\text{g_final}(V_{post}) - g}{\\text{temp_modifier}(T, Q_g_10, T_g_0) \\tau_g} \\\\
\\text{time_constant_B}(V) = \\tau_{B0} + a_B (1 - e^{λ_B V}) \\\\
\\text{time_constant_C}(V) = \\tau_{C0} + a_C (1 - e^{λ_C V}) \\\\
\\text{temp_modifier}(T, Q_10, T_0) = Q_10^{\\frac{T_0 - T}{10}} \\\\  
\\text{g_final}(V_{post}) = k (V_{post} - V_0) \\\\
I = (w_C * C + w_B * B - A) g (V_{post} - E) \\frac{1}{1 + \\frac{\\text{Mg}_O e^{- \\frac{z δ F V_{post}}{R T}}}{IC_50}}
```
where `V_pre` is the presynaptic membrane voltage, `V_post` is the postsynaptic membrane voltage, and `I` is the synaptic current that the receptor outputs to the postsynaptic neuron.

Arguments:
- `E` [mV, reversal potential]
- `k` [mV⁻¹, steepness of voltage-dependent change]
- `V_0` [mV, voltage value at which g_VD is equal to zero] 
- `τ_A` [ms, activation time constant]
- `τ_B0` [ms, fast deactivation time constant initial point]
- `τ_C0` [ms, slow deactivation time constant initial point]
- `τ_g` [ms, voltage-dependent gating time constant]
- `w_B` [weight of fast deactivation]
- `λ_B` [decay rate of fast deactivation time constant]
- `λ_C` [decay rate of slow deactivation time constant]
- `a_B` [tuning constant of fast deactivation time constant]
- `a_C` [tuning constant of slow deactivation time constant]
- `Mg_O` [mM, Mg concentration in the extracellular fluid]
- `IC_50` [mM, half-maximal inhibitory concentration at 0 mv]
- `T` [K, absolute temperature]
- `F` [C/mol, Faraday constant]
- `R` [J/(K*mol), Gas constant]
- `z` [Magnesium valence]
- `δ` [Magnesium constant]
- `Q_A_10` [temperature sensitivity of activation]
- `T_A_0` [Celsius, reference temperature of activation]
- `Q_B_10` [temperature sensitivity of fast deactivation]
- `T_B_0` [Celsius, reference temperature of fast deactivation]
- `Q_C_10` [temperature sensitivity of slow deactivation]
- `T_C_0` [Celsius, reference temperature of slow deactivation]
- `Q_g_10` [temperature sensitivity of conductance]
- `T_g_0` [Celsius, reference temperature of conductance]

References:
1. Moradi, K., Moradi, K., Ganjkhani, M. et al. (2013), A fast model of voltage-dependent NMDA receptors. J Comput Neurosci 34, 521–531

See also [`MoradiNMDAR`](@ref).
"""
@blox struct MoradiFullNMDAR(; name,
    namespace=nothing,
    E=-0.7, # mV
    k=0.007, # mV⁻¹
    V_0=-100, # mV
    τ_A=2.13,
    τ_B0=33.13,
    τ_C0=217.69,
    τ_g=1,
    w_B=0.65,
    λ_B=0.0243,
    λ_C=0.01,
    a_B=2.99,
    a_C=34.69,
    Mg_O=1, # mM
    IC_50=6.92, # mM
    T=37, # Celsius
    F=96485.332, # Faraday constant, C/mol
    R=8.314, # Gas constant, J/(K*mol)
    z_Mg=2, # Magnesium valence
    δ=0.99,
    Q_A_10=2.2,
    T_A_0=31.5, # Celsius
    Q_B_10=3.68,
    T_B_0=35, # Celsius
    Q_C_10=2.65,
    T_C_0=35, # Celsius
    Q_g_10=1.52,
    T_g_0=26, # Celsius
    w_C=1 - w_B,
    G_syn=3.0,
    V_shift=10.0,
    V_range=35.0,
    τ₁=0.1,
    spk_coeff=1
) <: AbstractReceptor
    @params(
        E, k, V_0, τ_A, τ_B0, τ_C0, τ_g, w_B, w_C, λ_B, λ_C,
        a_B, a_C, Mg_O, IC_50, T, F, R, z_Mg, δ = δ, Q_A_10, T_A_0,
        Q_B_10, T_B_0, Q_C_10, T_C_0, Q_g_10, T_g_0, G_syn,
        V_shift, V_range, τ₁, spk_coeff
    )
    @states(
        A = 0.0,
        B = 0.0,
        C = 0.0,
        g = 0.0,
        z = 0.0
    )
    @inputs V_pre = 0.0 V_post = 0.0
    @outputs
    @computed_properties_with_inputs I = (w_C * C + w_B * B - A) * g * (V_post - E) * (1 / (1 + Mg_O * exp(-z_Mg * δ * F * V_post / (R * T)) / IC_50))
    @equations begin
        @setup begin
            time_constant_B(V) = τ_B0 + a_B * (1 - exp(λ_B * V))
            time_constant_C(V) = τ_C0 + a_C * (1 - exp(λ_C * V))
            temp_modifier(T, Q_10, T_0) = Q_10^((T_0 - T) / 10)
            g_final(V) = k * (V - V_0)
        end
        D(A) = spk_coeff * z - A / (temp_modifier(T, Q_A_10, T_A_0) * τ_A)
        D(B) = spk_coeff * z - B / (temp_modifier(T, Q_B_10, T_B_0) * time_constant_B(V_post))
        D(C) = -C / (temp_modifier(T, Q_C_10, T_C_0) * time_constant_C(V_post))
        D(g) = (w_C * C + w_B * B) * (g_final(V_post) - g) / (temp_modifier(T, Q_g_10, T_g_0) * τ_g)
        D(z) = -z / τ₁ + (G_syn / (1 + exp(-4.394 * ((V_pre - V_shift) / V_range))))
    end
end

# Inherits namespace from the presynaptic neuron
function MoradiFullNMDAR(src::AbstractNeuron, dst::AbstractNeuron; name=:mFullNMDAR, namespace=full_namespaced_nameof(src), kwargs...)
    MoradiFullNMDAR(; name, namespace, kwargs...)
end

"""
    MoradiNMDAR(; E=-0.7, k=0.007, V_0=-100, g_VI=1, 
                τ_A=1.47, τ_B=391.64, τ_g=50, 
                Mg_O=1, IC_50=4.1, T=295.15, F=96485.332, R=8.314, 
                z=2, δ=0.8, spk_coeff=0.05)

An NMDA receptor model based on [1]. This is a simplified version of the full model [`MoradiFullNMDAR`](@ref) that does not include temperature correction terms and voltage-dependent time constants.

```math
\\frac{dA}{dt} = \\text{spk_coeff} z - \\frac{A}{\\tau_A} \\\\
\\frac{dB}{dt} = \\text{spk_coeff} z - \\frac{B}{\\tau_B} \\\\
\\frac{dg}{dt} = B \\frac{g_{VD}(V) - g}{\\tau_g} \\\\
g_{VD}(V) = k (V - V_0) \\\\
I = (B - A) (g_VI + g) (V - E) \\frac{1}{1 + \\frac{\\text{Mg}_O e^{- \\frac{z δ F V}{R T}}}{IC_50}}
```
where `V_pre` is the presynaptic membrane voltage and `V_post` is the postsynaptic membrane voltage,
`g` is a combination of a voltage-dependent and a voltage-independent conductance term, 
and `I` is the synaptic current that the receptor outputs to the postsynaptic neuron

Arguments:
- `E` [mV, reversal potential]
- `k` [mV⁻¹, steepness of voltage-dependent change]
- `V_0` [mV, voltage value at which g_VD is equal to zero] 
- `g_VI` [mS, voltage-independent conductance]
- `τ_A` [ms, activation time constant]
- `τ_B` [ms, deactivation time constant]
- `τ_g` [ms, voltage-dependent gating time constant],
- `Mg_O` [mM, Mg concentration in the extracellular fluid]
- `IC_50` [mM, half-maximal inhibitory concentration at 0 mv]
- `T` [K, absolute temperature]
- `F` [C/mol, Faraday constant]
- `R` [J/(K*mol), Gas constant]
- `z` [Magnesium valence]
- `δ` [Magnesium constant]
- `spk_coeff`: Spiking coefficient which multiplies incoming spiking activity. 
    It was tuned so that this simplified model matches the dynamics of the full `MoradiFullNMDAR` receptor model.

References:
1. Moradi, K., Moradi, K., Ganjkhani, M. et al. (2013), A fast model of voltage-dependent NMDA receptors. J Comput Neurosci 34, 521–531

See also [`MoradiFullNMDAR`](@ref).
"""
@blox struct MoradiNMDAR(; name,
    namespace=nothing,
    E=-0.7, # mV
    k=0.007, # mV⁻¹
    V_0=-100, # mV
    g_VI=1,
    τ_A=1.47,
    τ_B=391.64,
    τ_g=50,
    Mg_O=1, # mM
    IC_50=4.1, # mM
    T=295.15, # K
    F=96485.332, # Faraday constant, C/mol
    R=8.314, # Gas constant, J/(K*mol)
    z_Mg=2, # Magnesium valence
    δ=0.8,
    spk_coeff=0.05, # This was tuned by connecting two HHNeuronExci together
    G_syn=3.0,
    V_shift=10.0,
    V_range=35.0,
    τ₁=0.1
) <: AbstractReceptor
    @params(E, k, V_0, g_VI, τ_A, τ_B, τ_g, Mg_O, IC_50, T, F, R, z_Mg, δ, spk_coeff, G_syn, V_shift, V_range, τ₁)
    @states(
        A = 0.0,
        B = 0.0,
        g = 0.0,
        z = 0.0
    )
    @inputs V_pre = 0.0 V_post = 0.0
    @outputs
    @equations begin
        @setup begin
            g_VD(V) = k * (V - V_0)
        end
        D(A) = spk_coeff * z - A / τ_A
        D(B) = spk_coeff * z - B / τ_B
        D(g) = B * (g_VD(V_post) - g) / τ_g
        D(z) = -z / τ₁ + (G_syn / (1 + exp(-4.394 * ((V_pre - V_shift) / V_range))))
    end
    @computed_properties_with_inputs I = (B - A) * (g_VI + g) * (V_post - E) * (1 / (1 + Mg_O * exp(-z_Mg * δ * F * V_post / (R * T)) / IC_50))
end

# Inherits namespace from both the presynaptic and postsynaptic neuron. Cannot be re-used! (i.e. one receptor object per connection)
function MoradiNMDAR(src::AbstractNeuron, dst::AbstractNeuron;
    name=:mNMDAR,
    namespace=Symbol(full_namespaced_nameof(src), :_, full_namespaced_nameof(dst)),
    kwargs...)
    MoradiNMDAR(; name, namespace, kwargs...)
end

"""
    GABA_B_Synapse(; E_syn=-75, τ₁=200.1, τ₂=200, G_syn=0.007, V_shift=0, V_range=2, g=1)

A GABA B receptor model using damped oscillator dynamics. Equations and default parameter values are based on [1].
```math
\\frac{dG}{dt} = -\\frac{G}{τ_2} + z \\\\
\\frac{dz}{dt} = -\\frac{z}{τ_1} + \\frac{G_\\text{syn}}{1 + e^{-4.394(\\frac{V-V_\\text{shift}}{V_\\text{range}})}}
```
Arguments : 
- `E_syn` [mV, reversal potential]
- `G_syn` [mV, receptor conductance]
- `V_shift` [mV, transmitter threshold]
- `V_range` [mV, transmitter sensitivity]
- `τ₁` [ms, decay timescale for receptor conductance]
- `τ₂` [ms, decay timescale for receptor conductance]
- `g` [conductance gain]

See also [`GABA_A_Synapse`](@ref)

References : 
1. Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1998). Kinetic models of synaptic transmission. Methods in neuronal modeling, 2, 1-25.
"""
@blox struct GABA_B_Synapse(; name, namespace=nothing,
    E_syn=-75.0, τ₁=200.1, τ₂=200.0, G_syn=0.007,
    V_shift=0.0, V_range=2.0, g=1.0) <: AbstractReceptor
    @params E_syn τ₁ τ₂ G_syn V_shift V_range g
    @states G = 0.0 z = 0.0
    @inputs V = 0.0
    @outputs G
    @equations begin
        D(G) = ((-1 / τ₂) * G + z)
        D(z) = -z / τ₁ + (G_syn / (1 + exp(-4.394 * ((V - V_shift) / V_range))))
    end
end

# Inherits namespace from the presynaptic neuron
function GABA_B_Synapse(src::AbstractNeuron, ::AbstractNeuron;
    name=:gaba_b, namespace=full_namespaced_nameof(src), kwargs...)
    GABA_B_Synapse(; name, namespace, kwargs...)
end


"""
    NMDA_Synapse(; E_syn=0, τ=80, G_syn=0.2, V_shift=-20, V_range=2, g=1)

An NMDA receptor model using damped oscillator dynamics. Equations and default parameter values are based on [1].
```math
\\frac{dG}{dt} = \\frac{G_\\text{syn}}{1 + e^{-4.394(\\frac{V-V_\\text{shift}}{V_\\text{range}})}} -\\frac{G}{τ} \\\\
```
Arguments : 
- `E_syn` [mV, reversal potential]
- `G_syn` [mV, receptor conductance]
- `V_shift` [mV, transmitter threshold]
- `V_range` [mV, transmitter sensitivity]
- `τ` [ms, decay timescale for receptor conductance]
- `g` [conductance gain]

See also [`MoradiNMDAR`](@ref)

References : 
1. Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1998). Kinetic models of synaptic transmission. Methods in neuronal modeling, 2, 1-25.
"""
@blox struct NMDA_Synapse(; name, namespace=nothing,
    E_syn=0, τ=80, G_syn=0.2,
    V_shift=-20, V_range=2, g=1) <: AbstractReceptor
    @params E_syn τ G_syn V_shift V_range g
    @states G = 0.0
    @inputs V = 0.0
    @outputs G
    @equations begin
        D(G) = (G_syn / (1 + exp(-4.394 * ((V - V_shift) / V_range)))) - (1 / τ) * G
    end
end

# Inherits namespace from the presynaptic neuron
function NMDA_Synapse(src::AbstractNeuron, ::AbstractNeuron;
    name=:nmda, namespace=full_namespaced_nameof(src), kwargs...)
    NMDA_Synapse(; name, namespace, kwargs...)
end

"""
    MsnNMDAR(; E=-0.7, k=0.007, V_0=-100, g_VI=1,
                    τ_A=1.47, τ_B=391.64, τ_g=50,
                    Mg_O=1, IC_50=4.1, T=295.15, F=96485.332, R=8.314,
                    z_Mg=2, δ=0.8, spk_coeff=0.05,
                    G_syn=3.0, V_shift=10.0, V_range=35.0, τ₁=0.1)

MSN NMDA receptor model using the simplified Moradi-style kinetics, extended with a D1 modulation input.
D1 modulation is implemented as a multiplicative gain on the exported NMDA synaptic current:
`I = M_NMDA1 * I_base`, consistent with dopamine modulation used in the MSN microcircuit model.

```math
\\frac{dA}{dt} = \\text{spk\\_coeff}\\,z - \\frac{A}{\\tau_A} \\\\
\\frac{dB}{dt} = \\text{spk\\_coeff}\\,z - \\frac{B}{\\tau_B} \\\\
\\frac{dg}{dt} = B\\,\\frac{g_{VD}(V_{post}) - g}{\\tau_g} \\\\
\\frac{dz}{dt} = -\\frac{z}{\\tau_1} + \\frac{G_{syn}}{1 + e^{-4.394\\left(\\frac{V_{pre}-V_{shift}}{V_{range}}\\right)}} \\\\
g_{VD}(V) = k\\,(V - V_0) \\\\
I = M_{NMDA1}\\,(B-A)(g_{VI}+g)(V_{post}-E)\\,\\frac{1}{1 + \\frac{\\mathrm{Mg}_O e^{-\\frac{z_{Mg}\\,\\delta\\,F\\,V_{post}}{R\\,T}}}{IC_{50}}}
```

Arguments:
- `E` [mV]: Reversal potential.
- `k` [mV⁻¹]: Slope of the voltage-dependent conductance term `g_VD(V)`.
- `V_0` [mV]: Voltage offset in `g_VD(V)`.
- `g_VI` [-]: Voltage-independent conductance offset used as `(g_VI + g)`.
- `τ_A` [ms]: Time constant for state `A`.
- `τ_B` [ms]: Time constant for state `B`.
- `τ_g` [ms]: Time constant for voltage-dependent conductance state `g`.
- `τ_M` [ms]: Time constant for tracking `M_NMDA1` into the synaptic current output.
- `Mg_O` [mM]: Extracellular magnesium concentration.
- `IC_50` [mM]: Half-inhibition constant for Mg block at 0 mV.
- `T` [K]: Absolute temperature for the Mg block term.
- `F` [C/mol]: Faraday constant.
- `R` [J/(K*mol)]: Gas constant.
- `z_Mg` [-]: Magnesium valence (typically 2).
- `δ` [-]: Dimensionless Mg block factor.
- `spk_coeff` [-]: Gain applied to the presynaptic release proxy `z`.
- `G_syn` [-]: Release strength in the presynaptic logistic transmitter proxy.
- `V_shift` [mV]: Presynaptic voltage shift in the logistic transmitter proxy.
- `V_range` [mV]: Presynaptic voltage range in the logistic transmitter proxy.
- `τ₁` [ms]: Decay time constant of the presynaptic release proxy `z`.

Inputs:
- `V_pre` [mV]: Presynaptic membrane voltage (drives transmitter release proxy `z`).
- `V_post` [mV]: Postsynaptic membrane voltage (drives `g_VD` and Mg block).
- `M_NMDA1` [-]: D1 modulation factor (e.g., `1 + β1*ϕ1`). Default `1.0` means no modulation.

Computed properties:
- `I`: D1-modulated NMDA synaptic current.

References:
1. Humphries, M. D., Lepora, N., Wood, R., & Gurney, K. (2009). Dopamine-modulated dynamic cell assemblies generated by the GABAergic striatal microcircuit. *Frontiers in Computational Neuroscience*.
"""
@blox struct MsnNMDAR(; name,
    namespace=nothing,
    E=-0.7,        # mV
    k=0.007,       # mV⁻¹
    V_0=-100,      # mV
    g_VI=1,
    τ_A=1.47,
    τ_B=391.64,
    τ_g=50,
    τ_M=1.0,
    Mg_O=1,        # mM
    IC_50=4.1,     # mM
    T=295.15,      # K
    F=96485.332,   # C/mol
    R=8.314,       # J/(K*mol)
    z_Mg=2,        # Mg valence
    δ=0.8,
    spk_coeff=0.05,
    G_syn=3.0,
    V_shift=10.0,
    V_range=35.0,
    τ₁=0.1
) <: AbstractReceptor
    @params(E, k, V_0, g_VI, τ_A, τ_B, τ_g, τ_M, Mg_O, IC_50, T, F, R, z_Mg, δ,
        spk_coeff, G_syn, V_shift, V_range, τ₁)

    @states(
        A = 0.0,
        B = 0.0,
        g = 0.0,
        z = 0.0,
        M_NMDA1_state = 1.0
    )

    @inputs V_pre = 0.0 V_post = 0.0 M_NMDA1 = 1.0
    @outputs

    @equations begin
        @setup begin
            g_VD(V) = k * (V - V_0)
        end

        D(A) = spk_coeff * z - A / τ_A
        D(B) = spk_coeff * z - B / τ_B
        D(g) = B * (g_VD(V_post) - g) / τ_g

        D(z) = -z / τ₁ + (G_syn / (1 + exp(-4.394 * ((V_pre - V_shift) / V_range))))

        # Track D1 modulation input so it can be used in output connections.
        D(M_NMDA1_state) = (M_NMDA1 - M_NMDA1_state) / τ_M
    end

    @computed_properties_with_inputs begin
        I_base =
            (B - A) * (g_VI + g) * (V_post - E) *
            (1 / (1 + Mg_O * exp(-z_Mg * δ * F * V_post / (R * T)) / IC_50))

        I = M_NMDA1_state * (
            (B - A) * (g_VI + g) * (V_post - E) *
            (1 / (1 + Mg_O * exp(-z_Mg * δ * F * V_post / (R * T)) / IC_50))
        )
    end
end


"""
    MsnAMPAR(; E_syn=0.0, τ₁=0.1, τ₂=5.0, g=1.0)

MSN AMPA synapse based on the base `Glu_AMPA_Synapse`, extended with a D2 modulation input.
D2 modulation is implemented as a multiplicative gain on the synaptic drive:
`G_asymp_eff = M_AMPA2 * G_asymp`. For this linear synapse, this is equivalent to scaling
the resulting AMPA conductance (and thus the downstream AMPA current computed in the neuron),
consistent with the multiplicative D2 attenuation used in the MSN microcircuit model.

```math
\\frac{dG}{dt} = -\\frac{G}{\\tau_2} + z \\\\
\\frac{dz}{dt} = -\\frac{z}{\\tau_1} + M_{AMPA2}\\,G_{asymp}
```

Arguments:
- `E_syn` [mV]: Reversal potential (kept for API consistency).
- `τ₁` [ms]: Time constant of the auxiliary variable `z`.
- `τ₂` [ms]: Decay time constant of `G`.
- `g` [-]: Conductance gain (kept for API consistency; not used directly in this gate-only synapse).

Inputs:
- `G_asymp` [-]: Presynaptic synaptic drive / conductance target.
- `M_AMPA2` [-]: D2 modulation factor (e.g., `1 - β2*ϕ2`). Default `1.0` means no modulation.

Outputs:
- `G`: AMPA conductance-like gate variable.

References:
1. Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1998). Kinetic models of synaptic transmission. *Methods in Neuronal Modeling* (2nd ed.).
2. Humphries, M. D., Lepora, N., Wood, R., & Gurney, K. (2009). Dopamine-modulated dynamic cell assemblies generated by the GABAergic striatal microcircuit. *Frontiers in Computational Neuroscience*.
"""
@blox struct MsnAMPAR(; name,
    namespace=nothing,
    E_syn=0.0,
    τ₁=0.1,
    τ₂=5.0,
    g=1.0) <: AbstractReceptor
    @params E_syn τ₁ τ₂ g
    @states G = 0.0 z = 0.0

    @inputs G_asymp = 0.0 M_AMPA2 = 1.0
    @outputs G

    @equations begin
        D(G) = -G / τ₂ + z
        D(z) = -z / τ₁ + (M_AMPA2 * G_asymp)
    end
end


"""
    MsnD1Receptor(; K_D1=0.3, n_D1=1.0, τ_ϕ1=100.0, β1=0.5)

A minimal dopamine D1 receptor occupancy / modulation module for MSN models.
This blox converts a dopamine level `DA(t)` into a slow D1 activation state `ϕ1(t)` using Hill
steady-state occupancy with first-order relaxation, and exports a multiplicative NMDA gain
`M_NMDA1` that can be wired into an NMDA receptor blox.

```math
\\phi_{1,\\infty}(DA) = \\frac{DA^{n_{D1}}}{DA^{n_{D1}} + K_{D1}^{n_{D1}}} \\\\
\\frac{d\\phi_1}{dt} = \\frac{\\phi_{1,\\infty}(DA) - \\phi_1}{\\tau_{\\phi 1}} \\\\
M_{NMDA1} = 1 + \\beta_1 \\phi_1
```

Arguments:
- `K_D1` [-]: Half-saturation dopamine level for D1 (Hill constant).
- `n_D1` [-]: Hill coefficient for D1 occupancy.
- `τ_ϕ1` [ms]: Time constant for D1 activation dynamics.
- `β1` [-]: Maximal multiplicative gain of D1 on NMDA; output is `1 + β1*ϕ1`.

Inputs:
- `DA` [-]: Dopamine level (arbitrary units; must be consistent with `K_D1`).

Outputs:
- `M_NMDA1` [-]: D1 modulation factor for NMDA receptors.

References:
1. Humphries, M. D., Lepora, N., Wood, R., & Gurney, K. (2009). Dopamine-modulated dynamic cell assemblies generated by the GABAergic striatal microcircuit. *Frontiers in Computational Neuroscience*.
"""
@blox struct MsnD1Receptor(; name,
    namespace=nothing,
    K_D1=0.3,     # K_{D1}: half-saturation DA level for D1 (dimensionless)
    n_D1=1.0,     # n_{D1}: Hill coefficient
    τ_ϕ1=100.0,   # τ_{ϕ1}: time constant [ms] for D1 activation dynamics
    β1=0.5        # β1: maximal gain of D1 on NMDA
) <: AbstractReceptor
    # Parameters correspond to K_{D1}, n_{D1}, τ_{ϕ1}, β1 in the paper-style notation
    @params(K_D1, n_D1, τ_ϕ1, β1)

    # State variable: ϕ1(t), fraction of activated D1 receptors (0..1)
    @states(
        ϕ1 = 0.0
    )

    # Input: DA(t) = dopamine concentration / level (arbitrary units, e.g. 0..1)
    @inputs DA = 0.0

    # Explicit output port: D1 modulation factor on NMDA
    @outputs M_NMDA1

    @equations begin
        @setup begin
            # Steady-state D1 occupancy ϕ_{1,∞}(DA) given by a Hill function
            ϕ1_inf(DA) = (DA^n_D1) / (DA^n_D1 + K_D1^n_D1)
        end

        # First-order activation dynamics:
        # τ_{ϕ1} dϕ1/dt = ϕ_{1,∞}(DA) - ϕ1
        D(ϕ1) = (ϕ1_inf(DA) - ϕ1) / τ_ϕ1
    end
    @computed_properties begin
        # M_NMDA1 = 1 + β1 * ϕ1
        M_NMDA1 = 1.0 + β1 * ϕ1
    end
end

"""
    MsnD2Receptor(; K_D2=0.3, n_D2=1.0, τ_ϕ2=100.0, β2=0.3)

A minimal dopamine D2 receptor occupancy / modulation module for MSN models.
This blox converts dopamine level `DA(t)` into a slow D2 activation state `ϕ2(t)` using Hill
steady-state occupancy with first-order relaxation, and exports a multiplicative AMPA gain
`M_AMPA2` (typically an attenuation) that can be wired into an AMPA receptor blox.

```math
\\phi_{2,\\infty}(DA) = \\frac{DA^{n_{D2}}}{DA^{n_{D2}} + K_{D2}^{n_{D2}}} \\\\
\\frac{d\\phi_2}{dt} = \\frac{\\phi_{2,\\infty}(DA) - \\phi_2}{\\tau_{\\phi 2}} \\\\
M_{AMPA2} = 1 - \\beta_2 \\phi_2
```

Arguments:
- `K_D2` [-]: Half-saturation dopamine level for D2 (Hill constant).
- `n_D2` [-]: Hill coefficient for D2 occupancy.
- `τ_ϕ2` [ms]: Time constant for D2 activation dynamics.
- `β2` [-]: Gain of D2 on the target receptor (AMPA in the referenced MSN microcircuit); output is `1 - β2*ϕ2`.

Inputs:
- `DA` [-]: Dopamine level (arbitrary units; must be consistent with `K_D2`).

Outputs:
- `M_AMPA2` [-]: D2 modulation factor for AMPA receptors.

References:
1. Humphries, M. D., Lepora, N., Wood, R., & Gurney, K. (2009). Dopamine-modulated dynamic cell assemblies generated by the GABAergic striatal microcircuit. *Frontiers in Computational Neuroscience*.
"""
@blox struct MsnD2Receptor(; name,
    namespace=nothing,
    K_D2=0.3,     # K_{D2}: half-saturation DA level for D2
    n_D2=1.0,     # n_{D2}: Hill coefficient
    τ_ϕ2=100.0,   # τ_{ϕ2}: time constant [ms] for D2 activation dynamics
    β2=0.3        # β2: gain of D2 on target receptor (AMPA in Humphries 2009)
) <: AbstractReceptor
    # Parameters correspond to K_{D2}, n_{D2}, τ_{ϕ2}, β2
    @params(K_D2, n_D2, τ_ϕ2, β2)

    # State variable: ϕ2(t), fraction of activated D2 receptors (0..1)
    @states(
        ϕ2 = 0.0
    )

    # Input: DA(t) = dopamine concentration / level
    @inputs DA = 0.0

    # Explicit output port: D2 modulation factor on AMPA (NOT NMDA)
    @outputs M_AMPA2

    @equations begin
        @setup begin
            # Steady-state D2 occupancy ϕ_{2,∞}(DA)
            ϕ2_inf(DA) = (DA^n_D2) / (DA^n_D2 + K_D2^n_D2)
        end

        # First-order activation dynamics:
        # τ_{ϕ2} dϕ2/dt = ϕ_{2,∞}(DA) - ϕ2
        D(ϕ2) = (ϕ2_inf(DA) - ϕ2) / τ_ϕ2
    end
    @computed_properties begin
        # M_AMPA2 = 1 - β2 * ϕ2
        M_AMPA2 = 1.0 - β2 * ϕ2
    end
end

"""
    HTR5(;)

A 5-HT / kinase-condition selector for the Baxter et al. (1999) Aplysia sensory neuron model.
This blox is intentionally *not* a biophysical receptor current model. Instead, it acts as a
mode switch that outputs binary (0/1) flags corresponding to the four experimental conditions
used in the paper: Control, PKA-only, PKC-only, and 5-HT (PKA+PKC). These outputs can be wired
into a neuron blox that implements the corresponding parameter “corner” values.

Mode mapping (recommended to hold constant during a simulation):
- `mode = 0` → Control        → (PKA, PKC) = (0, 0)
- `mode = 1` → PKA only       → (PKA, PKC) = (1, 0)
- `mode = 2` → PKC only       → (PKA, PKC) = (0, 1)
- `mode = 3` → 5-HT (PKA+PKC) → (PKA, PKC) = (1, 1)

Arguments:
- (none)

Inputs:
- `mode` [-]: Condition index (0..3). Non-integers are thresholded with piecewise `ifelse`.

Outputs:
- `PKA` [-]: Binary kinase flag (0/1).
- `PKC` [-]: Binary kinase flag (0/1).
- `CTRL` [-]: 1 if in control mode, else 0.
- `PKA_only` [-]: 1 if in PKA-only mode, else 0.
- `PKC_only` [-]: 1 if in PKC-only mode, else 0.
- `HT_flag` [-]: 1 if in 5-HT mode, else 0.
- `mode_index` [-]: Debug index (0..3) after thresholding.

References:
1. Baxter, D. A., Byrne, J. H. (1999). Serotonergic modulation of two potassium currents in Aplysia sensory neurons. *Journal of Neurophysiology*, 82(6), 2914–2926. (doi:10.1152/jn.1999.82.6.2914)
"""
@blox struct HTR5(;
    name,
    namespace=nothing
) <: AbstractReceptor
    @params()

    # Keep the blox ODE-friendly: one dummy state with zero dynamics.
    @states(dummy = 0.0)

    @inputs mode = 0.0
    # mode meaning (recommended to keep constant during one simulation):
    #   0.0 = control
    #   1.0 = PKA only
    #   2.0 = PKC only
    #   3.0 = 5-HT (PKA + PKC)

    # Explicit outputs so this can be wired cleanly using Neuroblox connections.
    @outputs PKA PKC CTRL PKA_only PKC_only HT_flag mode_index

    @equations begin
        # No internal dynamics needed; dummy state is constant.
        D(dummy) = 0.0
    end
    @computed_properties_with_inputs begin
        # Produce strict 0/1 flags using piecewise ifelse.
        # If mode is held constant (typical use), this is numerically stable.

        CTRL = ifelse(mode < 0.5, 1.0, 0.0)

        PKA_only = ifelse(mode >= 0.5,
            ifelse(mode < 1.5, 1.0, 0.0),
            0.0)

        PKC_only = ifelse(mode >= 1.5,
            ifelse(mode < 2.5, 1.0, 0.0),
            0.0)

        HT_flag = ifelse(mode >= 2.5, 1.0, 0.0)

        # These are the only two signals the neuron needs:
        #   control      -> (PKA,PKC) = (0,0)
        #   PKA only     -> (1,0)
        #   PKC only     -> (0,1)
        #   5-HT         -> (1,1)
        PKA = ifelse(mode < 0.5, 0.0,
            ifelse(mode < 1.5, 1.0,
                ifelse(mode < 2.5, 0.0, 1.0)))
        PKC = ifelse(mode < 1.5, 0.0, 1.0)

        # For debugging/plotting: 0=control, 1=PKA, 2=PKC, 3=5-HT
        mode_index = ifelse(mode < 0.5, 0.0,
            ifelse(mode < 1.5, 1.0,
                ifelse(mode < 2.5, 2.0, 3.0)))
    end
end

"""
    CaTRPM4R(; ḡ=0.1, E_CAN=0.0, K_d=87.0,
              τ_can=400.0, X_base=0.0, X_CCh=500.0,
              use_Vdep=1.0, V_fix=-60.0)

A TRPM4 / I_CAN receptor-current module with nanodomain Ca coupling.
TRPM4 is Ca-impermeable; its opening is regulated by Ca in a local nanodomain produced by
nearby Ca sources (e.g., VGCC). This blox models:
1) a nanodomain Ca signal `can(t)` driven by inward Ca current `I_Ca` and relaxing to `Ca_bulk`;
2) a two-state TRPM4 activation gate `m(t)` with voltage-dependent rates and Ca-dependent scaling
   of the forward rate; and
3) an Ohmic TRPM4 current output `I = ḡ * m * (V - E_CAN)`.

```math
\\frac{d\\,can}{dt} = -X \\, I_{Ca,inward} + \\frac{Ca_{bulk} - can}{\\tau_{can}} \\\\
\\alpha(V) = 0.0057\\,e^{0.0060V}, \\quad \\beta(V) = 0.033\\,e^{-0.019V} \\\\
\\alpha'(V,can) = \\alpha(V)\\frac{can}{can + K_d} \\\\
\\frac{dm}{dt} = \\alpha'(V,can)(1-m) - \\beta(V)m \\\\
I_{TRPM4} = \\bar{g}\\,m\\,(V - E_{CAN})
```

Arguments:
- `ḡ` [mS/cm²]: Maximal TRPM4 conductance (tunable).
- `E_CAN` [mV]: Reversal potential for CAN current (often near 0 mV; set here to 0 mV by default).
- `K_d` [µM]: Ca binding dissociation constant for TRPM4 activation (paper reports ~87 µM).
- `τ_can` [ms]: Nanodomain Ca relaxation time constant.
- `X_base` [-]: Coupling gain from inward Ca current to nanodomain Ca in control.
- `X_CCh` [-]: Coupling gain under CCh (carbachol-like) condition.
- `use_Vdep` [-]: 1 uses instantaneous `V` for gating; 0 clamps gating voltage to `V_fix`.
- `V_fix` [mV]: Clamp voltage for gating when `use_Vdep=0`.

Inputs:
- `V` [mV]: Postsynaptic membrane voltage.
- `I_Ca` [µA/cm²]: Ca influx current (proxy) from Ca sources (not TRPM4 current).
- `Ca_bulk` [µM]: Bulk/bath baseline Ca level for nanodomain relaxation.
- `CCh` [-]: 0/1 (or 0..1) condition flag that interpolates X between `X_base` and `X_CCh`.

Computed properties:
- `I` [µA/cm²]: TRPM4 / I_CAN current.
- `can_pos` [-]: max(can,0) for debugging/inspection.
- `m_open` [-]: Open probability (same as `m`).
- `Vm_gate` [mV]: Effective voltage used for gating (clamped or not).
- `X_used` [-]: Effective coupling gain used in the current condition.

References:
1. Combe, C. L., Canavier, C. C., & Gasparini, S. (2023). (TRPM4 / I_CAN in CA1 pyramidal cells). *eLife*. (doi:10.7554/eLife.84387)
"""
@blox struct CaTRPM4R(;
    name,
    namespace=nothing,

    # ============================================================
    # TRPM4 / ICAN current (Ohmic)
    #   I_TRPM4 = ḡ * m * (V - E_CAN)
    # Units chosen to match HH neuron current balance:
    #   ḡ : mS/cm^2
    #   V : mV
    #   I : µA/cm^2
    # ============================================================
    ḡ=0.1,          # mS/cm^2  [tunable]
    E_CAN=0.0,       # mV       (non-specific cation reversal ~ 0 mV)

    # ============================================================
    # Ca binding for TRPM4 (paper: Kd ≈ 87 µM)
    #   α' = α / (1 + Kd / can) = α * can/(can+Kd)
    # ============================================================
    K_d=87.0,        # µM

    # ============================================================
    # Nanodomain Ca dynamics driven by Ca current (proxy)
    #   dcan/dt = (-X * I_Ca_inward) + (Ca_bulk - can)/τ_can
    #
    # IMPORTANT biological meaning:
    #   - I_Ca is NOT TRPM4 current. TRPM4 is Ca-impermeable.
    #   - I_Ca is the total Ca influx current (or a proxy) from
    #     co-localized Ca sources (e.g., VGCC) on the postsynaptic membrane.
    #   - X is a fitted coupling gain; paper uses X=0 (control) vs X=500 (CCh).
    #
    # Sign convention:
    #   If inward Ca current is negative (common I = g*(V-E)),
    #   then -I_Ca_inward(I_Ca) is positive and increases can.
    # ============================================================
    τ_can=400.0,     # ms
    X_base=0.0,      # control coupling
    X_CCh=500.0,    # coupling under CCh

    # ============================================================
    # Optional: evaluate gating kinetics at a clamped voltage
    # ============================================================
    use_Vdep=1.0,    # 1.0 = use V; 0.0 = clamp to V_fix for kinetics
    V_fix=-60.0      # mV
) <: AbstractReceptor

    @params(ḡ, E_CAN, K_d, τ_can, X_base, X_CCh, use_Vdep, V_fix)

    # States:
    #   can : nanodomain Ca2+ signal near TRPM4 (µM)
    #   m   : TRPM4 activation/open probability (0..1)
    @states(
        can = 0.0,
        m = 0.0
    )

    # Inputs:
    #   V       : postsynaptic membrane voltage (mV)
    #   I_Ca    : Ca influx current (or proxy) from postsynaptic Ca sources (µA/cm^2)
    #   Ca_bulk : bulk/bath baseline Ca (µM) (you can hold it constant externally)
    #   CCh     : 0/1 (or 0..1) flag for carbachol-like condition (sets X)
    @inputs V = 0.0 I_Ca = 0.0 Ca_bulk = 0.05 CCh = 0.0

    @outputs

    @equations begin
        @setup begin
            # -------- Voltage used by TRPM4 gating (optional clamp) --------
            V_eff(V, use_Vdep, V_fix) = use_Vdep * V + (1.0 - use_Vdep) * V_fix

            # -------- Paper TRPM4 voltage-dependent rates --------
            α(Vm) = 0.0057 * exp(0.0060 * Vm)
            β(Vm) = 0.033 * exp(-0.019 * Vm)

            # -------- Ca-dependent scaling of forward rate --------
            αprime(Vm, can_val, Kd) = α(Vm) * (can_val / (can_val + Kd))

            # -------- Effective nanodomain coupling X under CCh --------
            X_eff(Xb, Xcch, CCh) = Xb * (1.0 - CCh) + Xcch * CCh

            # -------- Only inward Ca current contributes to raising Ca --------
            # inward is negative under the usual sign convention
            I_Ca_inward(Ica) = min(Ica, 0.0)
        end

        # ============================================================
        # Nanodomain Ca dynamics
        # ============================================================
        D(can) = (-X_eff(X_base, X_CCh, CCh) * I_Ca_inward(I_Ca)) +
                 (Ca_bulk - can) / τ_can

        # ============================================================
        # TRPM4 gating kinetics (two-state form)
        #   dm/dt = α'(V,can)*(1-m) - β(V)*m
        # ============================================================
        D(m) = αprime(V_eff(V, use_Vdep, V_fix), max(can, 0.0), K_d) * (1.0 - m) -
               β(V_eff(V, use_Vdep, V_fix)) * m

    end

    @computed_properties_with_inputs begin
        # Ohmic TRPM4/ICAN current output
        I = ḡ * m * (V - E_CAN)

        # Debug/inspection helpers
        can_pos = max(can, 0.0)
        m_open = m
        Vm_gate = V_eff(V, use_Vdep, V_fix)
        X_used = X_base * (1.0 - CCh) + X_CCh * CCh
    end
end





"""
    Alpha7ERnAChR(; Ḡ_α7=0.001, Ca_o=2.0, RTF_Ca=13.32,
                  α=0.1, α1=0.01, β=5.0, γ=0.1, d2=5.0, d3=5.0,
                  ACh_baseline=0.0,
                  use_PAM=0.0, β_PAM=0.1, d3_PAM=0.1,
                  v_IP3R_s=170.0, v_RyR_s=57.0, v_leak_s=0.0001, v_SERCA_s=0.3,
                  K_IP3=1e-4, K_act=3e-4, K_inact=2e-4, K_RyR=1e-4, K_SERCA=1e-4,
                  ρ_ER=0.185,
                  τ_IP3=200.0, IP3_rest=0.0, k_IP3_from_Ca=0.0)

An α7 nicotinic ACh receptor model coupled to an intracellular ER Ca store module.
This blox combines:
1) α7 receptor gating via a 3-state Markov scheme (C, O, D) driven by ACh with optional PAM
   (positive allosteric modulator) that reduces exit rates from the open state; and
2) ER Ca handling with IP3R, RyR, leak, and SERCA fluxes that contribute a net ER→cytosol Ca flux
   `J_ER` to be added into the postsynaptic neuron’s cytosolic Ca equation.

The exported α7 current is treated as Ca-only:
```math
I_{\\alpha7} = \\bar{G}_{\\alpha7}\\,O\\,(V_m - E_{Ca}), \\quad
E_{Ca} = RTF_{Ca}\\ln\\left(\\frac{Ca_o}{Ca_i}\\right)
```

ER flux bookkeeping (positive ER→cytosol, negative uptake):
```math
J_{ER} = J_{IP3R} + J_{RyR} + J_{leak} - J_{SERCA}
```

Arguments (α7 gating):
- `Ḡ_α7` [µA/(mV·cm²)]: Maximal α7 conductance density (current units compatible with neuron balance).
- `Ca_o` [mM]: Extracellular Ca concentration (typically fixed).
- `RTF_Ca` [mV]: Nernst factor for divalent ions (≈ 13.32 mV at ~37°C).
- `α` [(µM⁻¹ ms⁻¹)]: ACh-dependent C→O rate coefficient (assumes ACh in µM).
- `α1` [ms⁻¹]: C→D desensitization rate.
- `β` [ms⁻¹]: O→C closing rate.
- `γ` [ms⁻¹]: D→O recovery rate.
- `d2` [ms⁻¹]: D→C recovery rate.
- `d3` [ms⁻¹]: O→D desensitization rate.
- `ACh_baseline` [µM]: Baseline ACh level added to the `ACh` input.
- `use_PAM` [-]: If 1, enable PAM effect using `β_PAM` and `d3_PAM`.
- `β_PAM` [ms⁻¹]: PAM-modified closing rate (used when PAM is active).
- `d3_PAM` [ms⁻¹]: PAM-modified desensitization rate from O (used when PAM is active).

Arguments (ER store / IP3):
- `v_IP3R_s, v_RyR_s, v_leak_s, v_SERCA_s` [s⁻¹]: Maximal flux rates (internally converted to ms⁻¹).
- `K_IP3, K_act, K_inact, K_RyR, K_SERCA` [mM]: Half-saturation constants (stored in mM).
- `ρ_ER` [-]: ER/cyt volume scaling factor used in the Ca_ER ODE.
- `τ_IP3` [ms]: Time constant for IP3 relaxation.
- `IP3_rest` [mM]: Rest (clamp) value of IP3.
- `k_IP3_from_Ca` [(mM/ms)/mM]: Optional coupling from cytosolic Ca to IP3 production.

Inputs:
- `V_m` [mV]: Membrane potential.
- `Ca_i` [mM]: Cytosolic Ca concentration (owned by the postsynaptic neuron).
- `ACh` [µM]: Phasic cholinergic input (added to `ACh_baseline`).
- `PAM` [-]: External PAM flag (0/1) that can override/enable PAM modulation.

Outputs:
- `I_α7` [µA/cm²]: α7 Ca current.
- `J_ER` [mM/ms]: Net ER→cytosol Ca flux contribution to add to neuron Ca_i dynamics.
- `O_open` [-]: Open-state fraction `O` (debug).
- `Ca_ER_out` [mM]: ER Ca store concentration (debug).
- `IP3_out` [mM]: IP3 level (debug).
- `h_IP3_out` [-]: IP3R inactivation gate (debug).
- J_IP3R, J_RyR, J_leak, J_SERCA: Optional debug flux components if wired/used downstream.

References:
1. King, J. R., et al. (2017). (α7 nAChR / Ca signaling framework used here). *Molecular Pharmacology*. (doi:10.1124/mol.117.111401)
"""
@blox struct Alpha7ERnAChR(;
    name,
    namespace=nothing,

    # ============================================================
    # α7 current (treated as Ca-only current)
    #   I_α7 = Ḡ_α7 * O * (V_m - E_Ca(Ca_o, Ca_i))
    # Units: consistent with TRN neuron currents (μA/(mV·cm^2) * mV = μA/cm^2)
    # ============================================================
    Ḡ_α7=0.001,            # μA/(mV·cm^2)
    Ca_o=2.0,              # mM (extracellular Ca, constant in TRN equations)
    RTF_Ca=13.32,           # mV, for z=2 Nernst factor at ~37°C

    # ============================================================
    # α7 Markov gating (C, O, D), rates
    # ============================================================
    α=0.1,                 # (μM^-1 ms^-1) if ACh is in μM (keep your paper convention)
    α1=0.01,                # ms^-1
    β=5.0,                 # ms^-1
    γ=0.1,                 # ms^-1
    d2=5.0,                 # ms^-1
    d3=5.0,                 # ms^-1

    # ACh drive (μM): baseline + input
    ACh_baseline=0.0,

    # Optional PAM modulation (reduce exit rates from open state)
    use_PAM=0.0,            # 0/1
    β_PAM=0.1,             # ms^-1
    d3_PAM=0.1,             # ms^-1

    # ============================================================
    # ER store flux parameters (rates given in s^-1 in King table -> convert to ms^-1)
    # Ca units in THIS BLOX: mM (to match TRN neuron Ca_i in mM)
    # ============================================================
    v_IP3R_s=170.0,        # s^-1
    v_RyR_s=57.0,         # s^-1
    v_leak_s=0.0001,       # s^-1
    v_SERCA_s=0.3,          # s^-1

    # Half-saturation constants (originally often reported in μM; here stored in mM)
    K_IP3=1e-4,           # 0.1 μM  -> 1e-4 mM
    K_act=3e-4,           # 0.3 μM  -> 3e-4 mM
    K_inact=2e-4,           # 0.2 μM  -> 2e-4 mM
    K_RyR=1e-4,           # 0.1 μM  -> 1e-4 mM
    K_SERCA=1e-4,           # 0.1 μM  -> 1e-4 mM

    # ER/cyt volume scaling
    ρ_ER=0.185,             # dimensionless

    # IP3 dynamics (if paper clamps IP3, set τ_IP3 very small and IP3_rest to clamp value)
    τ_IP3=200.0,            # ms
    IP3_rest=0.0,           # mM
    k_IP3_from_Ca=0.0       # (mM/ms)/mM  (optional coupling)
) <: AbstractReceptor

    @params(
        Ḡ_α7, Ca_o, RTF_Ca,
        α, α1, β, γ, d2, d3,
        ACh_baseline,
        use_PAM, β_PAM, d3_PAM,
        v_IP3R_s, v_RyR_s, v_leak_s, v_SERCA_s,
        K_IP3, K_act, K_inact, K_RyR, K_SERCA,
        ρ_ER,
        τ_IP3, IP3_rest, k_IP3_from_Ca
    )

    @states(
        # Markov states (fractions)
        C = 1.0,
        O = 0.0,
        D = 0.0,

        # ER store state (mM)
        Ca_ER = 1.0,      # mM (King table often uses 1 mM scale; adjust if your table differs)

        # IP3 / IP3R inactivation
        IP3 = 0.0,      # mM
        h_IP3 = 1.0       # 0..1
    )

    # Inputs:
    #   V_m : membrane potential (mV)
    #   Ca_i: cytosolic Ca (mM) -- owned by TRN neuron
    #   ACh : cholinergic transient (μM)
    #   PAM : 0/1 override
    @inputs V_m = -70.0 Ca_i = 2.4e-4 ACh = 0.0 PAM = 0.0

    # Outputs:
    #   I_α7 : α7 Ca current (μA/cm^2)
    #   J_ER : net ER -> cyt flux contribution to Ca_i (mM/ms)
    #   (and debug terms)
    @outputs I_α7 J_ER O_open Ca_ER_out IP3_out h_IP3_out J_IP3R J_RyR J_leak J_SERCA

    @equations begin
        @setup begin
            # ---- helpers ----
            per_ms(x_per_s) = x_per_s / 1000.0

            # PAM selection
            PAM_eff(use_PAM, PAM) = max(use_PAM, PAM)
            β_eff(β, β_PAM, pam) = pam > 0.5 ? β_PAM : β
            d3_eff(d3, d3_PAM, pam) = pam > 0.5 ? d3_PAM : d3

            # ACh effective
            ACh_eff(ACh_baseline, ACh) = max(ACh_baseline + ACh, 0.0)

            # IP3R factors
            f_IP3(IP3, K_IP3) = IP3 / (IP3 + K_IP3)
            f_act(Ca, K_act) = Ca / (Ca + K_act)

            # RyR activation
            f_RyR(Ca, K_RyR) = (Ca * Ca) / (Ca * Ca + K_RyR * K_RyR)

            # SERCA uptake
            f_SERCA(Ca, K_SERCA) = (Ca * Ca) / (Ca * Ca + K_SERCA * K_SERCA)

            # IP3R inactivation gate (simple)
            h_inf(Ca, K_inact) = K_inact / (Ca + K_inact)
            τ_h_IP3 = 100.0 # ms (if paper gives a different value, replace here)

            # ============================================================
            # ER fluxes (mM/ms)
            # Positive: ER -> cyt
            # ============================================================
        end

        # ============================================================
        # Markov gating: C <-> O, C <-> D, O <-> D
        # ============================================================
        D(C) = -(α * ACh_eff(ACh_baseline, ACh)) * C +
               β_eff(β, β_PAM, PAM_eff(use_PAM, PAM)) * O -
               α1 * C + d2 * D

        D(O) = (α * ACh_eff(ACh_baseline, ACh)) * C -
               (β_eff(β, β_PAM, PAM_eff(use_PAM, PAM)) +
                d3_eff(d3, d3_PAM, PAM_eff(use_PAM, PAM))) * O +
               γ * D

        D(D) = α1 * C +
               d3_eff(d3, d3_PAM, PAM_eff(use_PAM, PAM)) * O -
               (d2 + γ) * D

        # ============================================================
        # ER store ODE
        # ============================================================
        D(Ca_ER) = ρ_ER * (
            (v_SERCA_s / 1000.0) *
            f_SERCA(max(Ca_i, 0.0), K_SERCA) *
            Ca_i -
            ((v_IP3R_s / 1000.0) *
             (f_IP3(max(IP3, 0.0), K_IP3)^3) *
             (f_act(max(Ca_i, 0.0), K_act)^3) *
             (max(h_IP3, 0.0)^3) *
             (Ca_ER - Ca_i) +
             (v_RyR_s / 1000.0) *
             f_RyR(max(Ca_i, 0.0), K_RyR) *
             (Ca_ER - Ca_i) +
             (v_leak_s / 1000.0) *
             (Ca_ER - Ca_i))
        )

        # ============================================================
        # IP3 dynamics
        # ============================================================
        D(IP3) = (IP3_rest - IP3 + k_IP3_from_Ca * (Ca_i - 2.4e-4)) / τ_IP3
        D(h_IP3) = (h_inf(max(Ca_i, 0.0), K_inact) - h_IP3) / τ_h_IP3
    end

    @computed_properties_with_inputs begin
        # α7 Ca current (μA/cm^2)
        I_α7 = Ḡ_α7 * O * (V_m - RTF_Ca * log(Ca_o / max(Ca_i, 1e-12)))

        # Net ER->cyt flux contribution to Ca_i (mM/ms)
        J_ER = (v_IP3R_s / 1000.0) *
               ((max(IP3, 0.0) / (max(IP3, 0.0) + K_IP3))^3) *
               ((max(Ca_i, 0.0) / (max(Ca_i, 0.0) + K_act))^3) *
               (max(h_IP3, 0.0)^3) *
               (Ca_ER - Ca_i) +
               (v_RyR_s / 1000.0) *
               ((max(Ca_i, 0.0) * max(Ca_i, 0.0)) /
                (max(Ca_i, 0.0) * max(Ca_i, 0.0) + K_RyR * K_RyR)) *
               (Ca_ER - Ca_i) +
               (v_leak_s / 1000.0) *
               (Ca_ER - Ca_i) -
               (v_SERCA_s / 1000.0) *
               ((max(Ca_i, 0.0) * max(Ca_i, 0.0)) /
                (max(Ca_i, 0.0) * max(Ca_i, 0.0) + K_SERCA * K_SERCA)) *
               Ca_i

        # debug
        O_open = O
        Ca_ER_out = Ca_ER
        IP3_out = IP3
        h_IP3_out = h_IP3
    end
end



"""
    MuscarinicR(; ḡ_NCM=1.0, E_NCM=0.0,
                  α_Ca=0.2, α_max=10.0, β_m=1.0, gate_exp=1.0,
                  τ_CaNCM=1333.0, Ca_min=1.0e-5, Ca_scale=1000.0,
                  τ_M=200.0, M_baseline=0.0,
                  f_Na=0.576923)

Muscarinic-activated non-specific cation current (I_NCM / I_NaNCM / I_KNCM) module with Ca-dependent
gating and slow muscarinic activation dynamics.

This blox is intended to be wired into an ion-conserving neuron (e.g., `MuscarinicNeuron`) that keeps
explicit Na/K concentration state variables. The module exports:
- the total membrane current `I_NCM` (added to the membrane current balance), and
- a Na/K apportionment of that same current (`I_NaNCM`, `I_KNCM`) for ion bookkeeping.

Biophysical assumptions:
- The NCM current is modeled as an Ohmic non-specific cation current (typically reversing near 0 mV).
- Channel opening is Ca-dependent via a two-state gate `m(t)`.
- Muscarinic drive `M(t)` is low-pass filtered into an effective activation `ϕ_M(t)`.

```math
\\frac{d\\,Ca_{pool}}{dt} = \\frac{\\max(Ca_i, Ca_{min}) - Ca_{pool}}{\\tau_{CaNCM}} \\\\
M_{eff} = \\mathrm{clip}_{[0,1]}(M_{baseline} + M) \\\\
\\frac{d\\,\\phi_M}{dt} = \\frac{M_{eff} - \\phi_M}{\\tau_M} \\\\
\\alpha_m(Ca_{pool}) = \\min\\bigl(\\alpha_{Ca}\\,(Ca_{pool}\\,Ca_{scale}),\\,\\alpha_{max}\\bigr) \\\\
\\frac{dm}{dt} = \\alpha_m(Ca_{pool})\\,(1-m) - \\beta_m\\,m \\\\
I_{NCM} = \\bar g_{NCM}\\,\\max(\\phi_M,0)\\,m^{\\mathrm{gate\\_exp}}\\,(V_m - E_{NCM}) \\\\
I_{NaNCM} = f_{Na}\\,I_{NCM}, \\qquad I_{KNCM} = (1-f_{Na})\\,I_{NCM}
```

Arguments:
- `ḡ_NCM` [µA/(mV·cm²)]: Maximal NCM conductance density (tunable).
- `E_NCM` [mV]: NCM reversal potential (often near 0 mV).
- `α_Ca` [-]: Gain for Ca-dependent opening rate `α_m`.
- `α_max` [ms⁻¹]: Upper bound (saturation) for `α_m`.
- `β_m` [ms⁻¹]: Ca-independent closing rate for the two-state gate.
- `gate_exp` [-]: Exponent applied to `m` in the current expression (often 1).
- `τ_CaNCM` [ms]: Time constant for low-pass Ca pool `Ca_pool`.
- `Ca_min` [mM]: Floor applied to Ca to avoid numerical issues at extremely low Ca.
- `Ca_scale` [-]: Scale mapping neuron `Ca_i` (mM) into the effective Ca scale used by `α_m`.
- `τ_M` [ms]: Time constant for muscarinic activation low-pass `ϕ_M`.
- `M_baseline` [mM]: Baseline muscarinic tone added to the input `M`.
- `f_Na` [-]: Fraction of `I_NCM` assigned to Na for ion bookkeeping (K fraction is `1-f_Na`).

Inputs:
- `V_m` [mV]: Postsynaptic membrane potential.
- `Ca_i` [mM]: Cytosolic calcium concentration (owned by the postsynaptic neuron).
- `M` [-]: Muscarinic drive (0..1 recommended; baseline is added internally).

Outputs:
- `I_NCM` [µA/cm²]: Total NCM membrane current.
- `I_NaNCM` [µA/cm²]: Na component of `I_NCM` for ion bookkeeping.
- `I_KNCM` [µA/cm²]: K component of `I_NCM` for ion bookkeeping.
- `m_open` [-]: Gate open probability (debug; equals `m`).
- `Ca_pool_out` [-]: Filtered Ca pool (debug; equals `Ca_pool`).
- `M_act` [-]: Filtered muscarinic activation (debug; equals `ϕ_M`).

References:
1. Fransen, E., Alonso, A. A., & Hasselmo, M. E. (2002). Simulations of the Role of the Muscarinic-Activated Calcium-Sensitive Nonspecific Cation Current INCM in Entorhinal Neuronal Activity during Delayed Matching Tasks. *Journal of Neuroscience*, 22(3), 1081-1097.
"""
@blox struct MuscarinicR(;
    name,
    namespace=nothing,

    # ---------- Current parameters ----------
    # NOTE: Tuned to ensure spiking in MuscarinicNeuron.
    # ḡ_NCM increased to 1.0 to ensure effective depolarization.
    ḡ_NCM=1.0,         # µA/(mV·cm²)
    E_NCM=0.0,        # mV (INCM reversal often modeled ~0 mV)

    # ---------- Ca-dependent gating (Traub-style) ----------
    # In the paper, α_m = α_Ca * [Ca]_µM.
    # We use Ca_scale=1000 for mM→µM conversion.
    # α_Ca increased to 0.2 (from 0.02) to compensate for low resting Ca levels.
    α_Ca=0.2,         # α_m = min(α_Ca * Ca_eff, α_max)  [units: µM⁻¹ ms⁻¹]
    α_max=10.0,        # saturation rate [ms⁻¹]
    β_m=1.0,         # Ca-independent closing rate [ms⁻¹]
    gate_exp=1.0,      # exponent on m (usually 1)

    # ---------- INCM Ca pool: slow low-pass driven by Ca_i ----------
    τ_CaNCM=1333.0,    # ms (1.333 s)
    Ca_min=1.0e-5,    # floor for numerical stability (mM)
    Ca_scale=1000.0,   # mM → µM conversion factor (Fransén et al. 2002 uses µM)

    # ---------- muscarinic activation low-pass ----------
    τ_M=200.0,         # ms (choose; can be increased if you want slower modulation)
    M_baseline=0.0,    # baseline muscarinic tone

    # ---------- Na/K apportionment ----------
    f_Na=0.576923      # Na fraction of I_NCM; K fraction = 1 - f_Na
) <: AbstractReceptor

    @params(ḡ_NCM, E_NCM,
        α_Ca, α_max, β_m, gate_exp,
        τ_CaNCM, Ca_min, Ca_scale,
        τ_M, M_baseline,
        f_Na)

    @states(
        Ca_pool = 0.0,    # filtered Ca pool (mM-like)
        ϕ_M = 0.0,    # filtered muscarinic activation
        m = 0.0     # INCM gate
    )

    @inputs V_m = -70.0 Ca_i = 2.4e-4 M = 0.0

    @outputs I_NCM I_NaNCM I_KNCM m_open Ca_pool_out M_act

    @equations begin
        @setup begin
            clamp01(x) = min(max(x, 0.0), 1.0)

            # effective muscarinic drive (baseline + input)
            M_eff(Mb, Min) = clamp01(Mb + Min)

            # Traub-style Ca-dependent opening rate
            αm(Ca_val, α_Ca, α_max, Ca_scale) = min(α_Ca * (Ca_val * Ca_scale), α_max)
        end

        # Low-pass Ca_i into an INCM-local Ca pool
        D(Ca_pool) = (max(Ca_i, Ca_min) - Ca_pool) / τ_CaNCM

        # Low-pass muscarinic activation
        D(ϕ_M) = (M_eff(M_baseline, M) - ϕ_M) / τ_M

        # Two-state gate: dm/dt = α(Ca)*(1-m) - β*m
        D(m) = αm(max(Ca_pool, Ca_min), α_Ca, α_max, Ca_scale) * (1.0 - m) - β_m * m
    end

    @computed_properties_with_inputs begin
        # total INCM/NCM current
        I_NCM = ḡ_NCM * max(ϕ_M, 0.0) * (m^(gate_exp)) * (V_m - E_NCM)

        # Na/K apportionment for ion bookkeeping
        I_NaNCM = f_Na * (ḡ_NCM * max(ϕ_M, 0.0) * (m^(gate_exp)) * (V_m - E_NCM))
        I_KNCM = (1.0 - f_Na) * (ḡ_NCM * max(ϕ_M, 0.0) * (m^(gate_exp)) * (V_m - E_NCM))

        # debug
        m_open = m
        Ca_pool_out = Ca_pool
        M_act = ϕ_M
    end
end


"""
    Beta2nAChR(; ḡ_ACh=5.0, E_ACh=0.0, τ_act=5.0, τ_des_base=500.0, ...)

β2-containing nicotinic acetylcholine receptor (nAChR) model for VTA neurons.

Models distinct effects of pulsatile ACh input versus tonic nicotine on receptor
activation and desensitization using a 4-state scheme.

```math
g_{eff} = \\bar{g}_{ACh} \\cdot ACh_{act} \\cdot (1 - ACh_{des})
```

Arguments:
- `ḡ_ACh` [mS/cm²]: Maximal conductance (5-10 for DA, 1.5-4 for GABA neurons).
- `E_ACh` [mV]: Reversal potential (0 mV for non-specific cation channel).
- `τ_act` [ms]: Activation time constant.
- `τ_des_base`, `τ_des_scale`, `K_t`: Desensitization time constant parameters.
- `EC50`, `IC50` [µM]: Half-maximal concentrations for activation/desensitization.
- `n_act`, `n_des` [-]: Hill coefficients.
- `w` [-]: Nicotine potency relative to ACh.

Inputs:
- `V` [mV]: Postsynaptic membrane voltage.
- `inp_ACh` [µM]: Pulsatile ACh input.
- `inp_Nic` [µM]: Tonic nicotine concentration.

Outputs:
- `I` [µA/cm²]: nAChR current (outward positive, g*(V-E)).

References:
1. Morozova, E. O., et al. (2020). Distinct Temporal Structure of Nicotinic ACh Receptor Activation Determines Responses of VTA Neurons to Endogenous ACh and Nicotine. *eNeuro*, 7(4). (doi:10.1523/ENEURO.0418-19.2020)
"""
@blox struct Beta2nAChR(;
    name,
    namespace=nothing,
    ḡ_ACh=5.0,
    E_ACh=0.0,
    τ_act=5.0,
    τ_des_base=500.0,
    τ_des_scale=16e5,
    K_t=0.5,
    EC50=1.0,
    IC50=0.1,
    n_act=1.05,
    n_des=0.5,
    w=3.0
) <: AbstractReceptor

    @params(ḡ_ACh, E_ACh, τ_act, τ_des_base, τ_des_scale, K_t, EC50, IC50, n_act, n_des, w)

    @states(
        ACh_act = 0.0,
        ACh_des = 0.0
    )

    @inputs V = -60.0 inp_ACh = 0.0 inp_Nic = 0.0

    @outputs I g_eff ACh_act_out ACh_des_out

    @equations begin
        @setup begin
            agonist_eff(ach, nic, w) = max(ach + w * nic, 1e-12)
            ACh_act_inf(ach, nic, w, ec50, n) = 1.0 / (1.0 + (ec50 / agonist_eff(ach, nic, w))^n)
            ACh_des_inf(nic, ic50, n) = ifelse(nic < 1e-12, 0.0, 1.0 / (1.0 + (ic50 / nic)^n))
            τ_des(nic, τ_base, τ_scale, kt) = max(τ_base - τ_scale / (1.0 + (max(nic, 1e-12) / kt)^3), 1.0)
        end

        D(ACh_act) = (ACh_act_inf(inp_ACh, inp_Nic, w, EC50, n_act) - ACh_act) / τ_act
        D(ACh_des) = (ACh_des_inf(inp_Nic, IC50, n_des) - ACh_des) / τ_des(inp_Nic, τ_des_base, τ_des_scale, K_t)
    end

    @computed_properties_with_inputs begin
        g_eff = ḡ_ACh * ACh_act * (1.0 - ACh_des)
        I = ḡ_ACh * ACh_act * (1.0 - ACh_des) * (V - E_ACh)
        ACh_act_out = ACh_act
        ACh_des_out = ACh_des
    end
end
