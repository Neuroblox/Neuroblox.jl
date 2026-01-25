# [Defining your own Blox with `@blox`](@id at_blox)

```julia
@blox struct StructName(; name, namespace, other_kwargs...) <: SuperType
    @params    ...
    @states    ...
    @inputs    ...
    @outputs   ...
    @equations ...
    [@noise_equations ...]
    [@discrete_events ...]
    [@event_times ...]
    [@computed_properties ...]
    [@computes_properties_with_inputs ...]
    [@extra_fields ...]
end
```

Define a Neuroblox-compatible struct ("blox") and automatically generates interface methods
for use with [GraphDynamics.jl](https://github.com/Neuroblox/GraphDynamics.jl) from Neuroblox.

# Overview

The `@blox` macro transforms a specialized `struct` declaration containing domain-specific
macros (e.g. `@params`, `@states`, `@inputs`, `@equations`, etc.) into a fully functional
type that can participate in dynamic graph simulations. It automatically generates:

- The underlying Julia `struct` type (a subtype of your chosen supertype, e.g. `AbstractBlox`)
- Standard interface definitions:
  - `Base.getproperty` and `Base.nameof`
  - `NeurobloxBase.param_symbols`, `state_symbols`, `input_symbols`, `output_symbols`
  - `GraphDynamics.to_subsystem`
  - `GraphDynamics.subsystem_differential`
  - `GraphDynamics.initialize_input`
  - `NeurobloxBase.outputs`
  - Optionally, `GraphDynamics.has_discrete_events`, `discrete_event_condition`, `apply_discrete_event!`
  - Optionally, `GraphDynamics.event_times`
  - Optionally, `GraphDynamics.isstochastic`, `apply_subsystem_noise!`
  - Optionally, `GraphDynamics.computed_properties` / `NeurobloxBase.computed`
  - Optionally, `GraphDynamics.computed_properties_with_inputs`
- A default constructor that initializes the block's parameters, states, inputs, and any
  additional user-defined fields.

# Constructor Signature

The struct header must be a valid constructor signature with keyword arguments:
```julia
@blox struct MyBlox(; name, namespace=nothing, param1=default1, ...) <: SuperType
```

The `name` and `namespace` arguments are mandatory and automatically stored. Additional
keyword arguments can be used to initialize parameters or extra fields.

# Body Directives

The body of the struct can contain the following directives:

## Mandatory Directives

### `@params`

Define model subsystem parameters. Parameters can have default values or be passed from
constructor arguments:
 
```julia
@params C Eₘ Rₘ τ θ E_syn G_syn=0.2 I_in
```

Parameters without defaults must be provided as keyword arguments in the constructor.
Parameters are automatically stored in a `param_vals` named tuple and accessible via
property access.

**Interface generated:** `NeurobloxBase.param_symbols(::Type{MyBlox})`

### `@states`

Define the model subsystem's differentiable state variables and their initial values.
Only define states here which are governed by a differential equation, not computed
states or inputs:

```julia
@states V=-70.0 G=0.0
```

All states must have initial values. States become properties of the subsystem and
are used in the `@equations` directive.

**Interface generated:** `NeurobloxBase.state_symbols(::Type{MyBlox})`

### `@inputs`

Define external input variables and their zero/initial values (used as the zero element
when summing all inputs to a blox):

```julia
@inputs jcn=0.0
```

or for multiple inputs:

```julia
@inputs(
    I_syn=0.0,
    I_in=0.0,
    I_asc=0.0,
    jcn=0.0,
)
```

Input symbols become parameters in the generated `subsystem_differential` function.

**Interface generated:** 
- `NeurobloxBase.input_symbols(::Type{MyBlox})`
- `GraphDynamics.initialize_input(::Subsystem{MyBlox})`

### `@outputs`

Declare which variables (typically states) should be exposed as outputs of the subsystem:

```julia
@outputs V G
```

Output symbols must correspond to defined states. These are used by GraphDynamics to
connect subsystems.

**Interface generated:** 
- `NeurobloxBase.output_symbols(::Type{MyBlox})`
- `NeurobloxBase.outputs(::Subsystem{MyBlox})`

### `@equations`

Define the differential equations governing state evolution. Each entry must be of the
form `D(state) = expression`:

```julia
@equations begin
    D(V) = (-(V-Eₘ)/Rₘ + I_in + jcn)/C
    D(G) = (-1/τ)*G
end
```

**Requirements:**
- Must be declared after `@params`, `@states`, and `@inputs`
- Must provide exactly one equation for each state variable
- Can reference parameters, states, inputs, and time `t` by name
- Can include `@setup` macro calls for e.g. defining functions and variables before the equations are calculated.

**Interface generated:** `GraphDynamics.subsystem_differential(::Subsystem{MyBlox}, inputs, t)`

## Optional Directives

### `@noise_equations`

Define stochastic noise terms for states, creating a system compatible with StochasticDiffEq.jl.
Each entry must be of the form `W(state) = expression`:

```julia
@noise_equations begin
    W(V) = ζ
    W(G) = σ * sqrt(G)
end
```

This defines diagonal noise where the noise term for state `V` is `ζ * dW` and for `G` is `σ * sqrt(G) * dW`,
where `dW` represents Wiener process increments.

**Requirements:**
- Must be declared after `@params`, `@states`, and `@inputs`
- Can only define noise for states that exist in `@states`
- Currently only supports diagonal noise (one noise term per state)
- Not all states need to have noise equations
- Can reference parameters, states, inputs, and time `t` by name
- Can include `@setup` macro calls like `@equations`

**Interface generated:**
- `GraphDynamics.isstochastic(::Type{MyBlox})` - returns `true`
- `GraphDynamics.apply_subsystem_noise!(v, ::Subsystem{MyBlox}, t)` - fills noise vector

### `@discrete_events`

Specify conditions and effects for discrete events which trigger during simulation.
Events fire when the condition becomes true and apply instantaneous updates to states
or parameters:

```julia
@discrete_events (V >= θ) => (V=Eₘ, G=G+G_syn)
```

Single assignment syntax is also supported:
```julia
@discrete_events (V >= θ) => V=Eₘ
```

**Limitations:**
- Currently only supports a single discrete event per blox.
- If you have an event that triggers at a certain time `t_event`, you need to supply that time to `@event_times`
**in addition** to checking if `t == t_event` in `@discrete_events`.

**Interface generated:**
- `GraphDynamics.has_discrete_events(::Type{MyBlox})`
- `GraphDynamics.discrete_event_condition(::Subsystem{MyBlox}, t, _)`
- `GraphDynamics.apply_discrete_event!(integrator, sview, pview, ::Subsystem{MyBlox}, _)`

For more complicated events, you may need to omit this directive and overload `has_discrete_events`, `discrete_event_condition` and `apply_discrete_event!` manually.

### `@event_times`

Specify predetermined times at which events should occur:

```julia
@event_times t_event
```

or for multiple times:
```julia
@event_times (t1, t2, t3)
```

The expression can reference parameters and states.

**Interface generated:** `GraphDynamics.event_times(::Subsystem{MyBlox})` which adds the times to the `tstops` kwarg in DifferentialEquations.jl solvers.

### `@extra_fields`

Add arbitrary extra fields to the struct (e.g., metadata, cached values, configuration):

```julia
@extra_fields default_synapses::Set{Any}=Set() cortical::Bool=true
```

Each entry must be an assignment with optional type annotation. Extra fields are
initialized in the constructor and stored as regular struct fields.

# Code in Constructor Body

Code can be placed directly in the struct body outside of the macro directives.
This code executes in the constructor before the struct is instantiated:

```julia
@blox struct MyBlox(; name, namespace=nothing, delayed=false) <: AbstractBlox
    if delayed
        error("Delay systems are currently not supported")
    end
    @params ...
    @states ...
    # ... rest of definition
end
```

This allows for validation, conditional logic, or computed initialization values.

## With Conditional Logic and Extra Fields
```julia
@blox struct JansenRit(; name, namespace=nothing, 
                       cortical=true, delayed=false) <: AbstractNeuralMass
    if delayed
        error("Delay systems are currently not supported")
    end
    τ = cortical ? 1 : 14
    @params τ
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @extra_fields region_type::Symbol=(cortical ? :cortical : :thalamic)
    @equations begin
        D(x) = y - (2x/τ)
        D(y) = -x/(τ^2) + jcn/τ
    end
end
```

# Examples

## Basic Linear System
```julia
@blox struct LinearNeuralMass(; name, namespace=nothing) <: AbstractNeuralMass
    @params
    @states x=0.0
    @inputs jcn=0.0
    @equations begin
        D(x) = jcn
    end
end
```

## Stochastic System
```julia
@blox struct NoisyOscillator(; name, namespace=nothing, 
                             ω=1.0, ζ=0.5, σ=0.1) <: AbstractNeuralMass
    @params ω ζ σ
    @states x=1.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = -(ω^2)*x - 2*ω*ζ*y + jcn
    end
    @noise_equations begin
        W(x) = σ
        W(y) = σ
    end
end
```

## Oscillator with Parameters
```julia
@blox struct HarmonicOscillator(; name, namespace=nothing,
                                ω=25*(2*pi)*0.001, ζ=1.0) <: AbstractNeuralMass
    @params ω ζ
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y - (2*ω*ζ*x)
        D(y) = -(ω^2)*x
    end
end
```

## Spiking Neuron with Discrete Events
```julia
@blox struct LIFNeuron(; name, namespace=nothing, 
                       C=1.0, θ=-50.0, Eₘ=-70.0) <: AbstractNeuron
    @params C θ Eₘ
    @states V=-70.0 G=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = (-(V-Eₘ) + jcn)/C
        D(G) = -G/10.0
    end
    @discrete_events (V >= θ) => (V=Eₘ, G=G+0.002)
end
```

## With Conditional Logic and Extra Fields
```julia
@blox struct JansenRit(; name, namespace=nothing, cortical=true, τ = cortical ? 1 : 14) <: AbstractNeuralMass
    @params τ
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @extra_fields region_type::Symbol=(cortical ? :cortical : :other)
    @equations begin
        D(x) = y - (2x/τ)
        D(y) = -x/(τ^2) + jcn/τ
    end
end
```

# Notes

- The macro validates that equations match defined states
- Parameters, states, and inputs are accessible as properties in equations
- The `namespace` field is used for hierarchical naming in larger systems
- Generated structs are immutable but the parameteters and extra fields can be mutable structures if needed.
