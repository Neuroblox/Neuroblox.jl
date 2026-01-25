# Define experiments using `@experiment`

Say you have a model of a brain circuit, and you would like to see its outputs with various sets of parameters. The @experiment macro provides an easy interface for iterating on a problem by changing its parameters, initial conditions, and timespan.

First let's create a circuit and then a simulation.

```julia
@graph g begin
    @nodes begin
        Brainstem = NGNMM_theta()
        Cortex = Cortex(
            N_wta=20, 
            N_exci=5, 
            density=0.05, 
            weight=1.0, 
            I_bg_ar=4, 
            G_syn_inhib=4.0, 
            τ_inhib=70, 
            G_syn_ff_inhib=1.5
        )
    end
    @connections begin
        Brainstem => Cortex, [weight = 20]
    end
end

tspan = (0., 500.)
prob = ODEProblem(g, [], tspan)
```

## Modifying parameters

Let's say we apply a drug whose known effect is to scale the timescale of the GABA synapses. Let's also say that this effect is limited to the cortex. Then we can write:

```julia
prob_drugged = @experiment prob begin
    @setup begin
        scale = 2.0
    end

    GABA_A_Synapse in Cortex, τ₂ -> τ₂ * scale
end
```

Inside the experiment macro, the `@setup` block is optionally used to define variables. 

The next line gives a rule for changing the parameters of the problem. This has two parts:

1. Blox selection: `GABA_A_Synapse in Cortex` is used to select all the `GABA_A_Synapse` inside the graph for `Cortex` (reference the API to see the full list of recognized selections)
2. Parameter modification: `τ₂ -> τ₂ * scale` sets the `τ₂` parameter of all of these synapses to its original value times `scale`. This should be read as a function.

Any number of these parameter modification lines may be included inside the macro.

Note also that `prob_drugged` is a copy of the original `prob`; if we wanted to simulate the original `prob`, we could.

## Modifying initial conditions and timespan

Let's see how to change initial conditions and timespans for the problem. Let's say we want to dose the drug only after an initial warmup period (that is, simulate the undrugged problem for some time t, then apply the drug effect). We also don't want the drugged simulation to run quite as long.

```julia
sol_undrugged = sol(prob, Vern7())

prob_drugged = @experiment prob begin
    @initial_conditions sol_undrugged.u[end]
    @tspan (0., 300.)
    @setup begin
        scale = 2.0
    end

    GABA_A_Synapse in Cortex, τ₂ -> τ₂ * scale
end
```
