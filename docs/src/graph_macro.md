# Build graphs using the `@graph` macro

```julia
@graph begin
    @nodes begin
        (...)
    end

    @connections begin
        (...)
    end
end
```
or
```julia
@graph name begin
    @nodes begin
        (...)
    end

    @connections begin
        (...)
    end
end
```

Build a `GraphSystem` from a list of nodes corresponding to blox and a set of connections between the nodes. 

Nodes are defined using normal julia assignment syntax, and connections are defined using pair syntax, followed by an optional list of keyword arguments for the connection.

```julia
@graph begin
    @nodes begin
        nn1 = HHNeuronExci(name=Symbol("nrn1"), I_bg=3)
        nn2 = HHNeuronExci(name=Symbol("nrn2"), I_bg=2)
        nn3 = HHNeuronInhib(name=Symbol("nrn3"), I_bg=2)
    end

    @connections begin
        nn1 => nn2, [weight = 1]
        nn2 => nn3, [weight = 1]
        nn3 => nn1, [weight = 0.2]
    end
end
```
The `name` kwarg of the nodes will be set equal to the variable they're assigned to, and their `namespace` equal to the `name` of the outer graph (default `nothing`).

One can insert normal Julia code outside of the `@nodes` and `@connections` blox to improve readability or perform other logic:

```julia
@graph begin
    μ_E = 1.5
    σ_E = 0.15
    μ_I = 0.8
    σ_I = 0.08

    @nodes begin
        ping_exci1 = PINGNeuronExci(I_ext = rand(rng, Normal(μ_E, σ_E)))
        ping_exci2 = PINGNeuronExci(I_ext = rand(rng, Normal(μ_E, σ_E)))
        ping_inhib = PINGNeuronInhib(I_ext = rand(rng, Normal(μ_I, σ_I)))
    end
end
```

# Defining nodes and connections in loops
Nodes and connections can also be defined programmatically using for loops and comprehensions. The syntax here is rigid:

The syntax for nodes is rigid: nodes must have an assignment still. For example, writing:
```julia
@nodes begin
    exci_neurons = [HHNeuronExci() for i in 1:10]

    inh_neurons = for i in 1:10
        HHNeuronInhib()
    end
end
```
will assign the array of 10 excitatory neurons to `exci_neurons` and the array of 10 inhibitory neurons to `inh_neurons`. (Note that this is different from how normal for loops behave.)

By default, the neurons inside will have the name of the array, followed by their index. For example, the first excitatory neuron will have the name `exci_neurons_1`.

Assignments are allowed inside for loops. For example, we can write
```julia
@nodes begin
    exci_neurons = [HHNeuronExci() for i in 1:10]

    inh_neurons = for i in 1:10
        name = "my_neuron$i"
        HHNeuronInhib(name = name)
    end
end
```
to manually pick the names assigned to the inhibitory neurons. However, the **last line of the for loop must be a blox constructor call.**

Edges and for loops can be defined using a comprehension as well.

```julia
@graph begin
    @nodes begin
        if_neurons = [IFNeuron(I_in = rand()) for _ in 1:2]
        qif_neuron = QIFNeuron(I_in = rand())
    end

    neurons = [if_neurons; qif_neuron]

    @connections begin
        for (n1, n2) in zip(neurons, neurons)
            n1 => n2, [weight = 1]
        end
    end
end
```
