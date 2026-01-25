# Defining your own connections with `@connection`
```julia
@connection function (conn::ConnType)(src::SrcType, dst::DstType, t)
    # optional assignments...

    @equations begin
        ...
    end

    # Optional: define discrete events
    @discrete_events begin 
        ...
    end
end
```

Define basic connections between blox and automatically generates interface methods for interface methods
for use with [GraphDynamics.jl](https://github.com/Neuroblox/GraphDynamics.jl) from Neuroblox.

# Overview
This macro takes the form of a callable struct (a function definition where the function is a struct). The struct should be a `ConnectionRule`, which include types like `BasicConnection`, `EventConnection`, and `ReverseConnection`:
- `conn`: The callable connection. This must have a type annotation
- `src`: The source blox. This argument must have a type specification.
- `dst`: The destination blox. This argument must have a type specification.
- `t`: The time argument

It is possible to use your own connection type, if you have defined the type beforehand: 
```julia
struct MyConnection
    weight::Float64
    extra_data
end

@connection function (conn::MyConnection)(...)
    ...
end
```

The code will generate:
- A method for the connection (e.g. `BasicConnection`) applied to your specified source and destination types, which returns the equations of the connection.

If discrete events are defined, the macro will aditionally generate the following:
- `GraphDynamics.event_times`
- `GraphDynamics.discrete_event_condition`
- `GraphDynamics.has_discrete_event`
- `GraphDynamics.apply_discrete_event!`

# Assignments in function body
Variable assignments can be inserted before the `@equations` and `@discrete_events` blocks. This is useful for making equations and discrete event definitions more legible, e.g. by defining `w = conn.weight` so one does not have to write `conn.weight` every time it is used in the definitions.

Note that fields of the connection, source, and destination are accessed using normal Julia syntax.

# Equations block
The `@equations` macro is used to define the equations of the connection. Each line in this block corresponds to a variable that is fed as an input into the destination blox.

```julia
@connection function (conn::BasicConnection)(src::KuramotoOscillator, dst::KuramotoOscillator, t)
    w = conn.w
    x₀ = src.θ
    xᵢ = dst.θ

    @equations begin
        jcn = w * sin(x₀ - xᵢ)
    end
end
```
In this case, `jcn` is fed as an input to the destination `KuramotoOscillator`. 

# Discrete Events
There are connections that have discrete event attached to them. Two macros are used to fully define discrete events for a connection: `@event_times` and `@discrete_events`.

### Declaring `@event_times`
`@event_times` is used to declare the times that the integrator must stop in order to trigger a discrete event. Event times should be explicitly declared for your connection (the default method simply returns `conn.event_times`, which may or may not exist for types other than `EventConnection`).

The `@event_times` macro takes a single argument that will become the function body of `GraphDynamics.event_times`. This argument can be a single number or an iterable.

For example, multiple event times can be declared as a vector:
```julia
@event_times [1, 2]
```

The assignments declared earlier are also accessible to the function body of `GraphDynamics.event_times`. So one can write: 

```julia
@connection (...)
    t_init = 0.1
    t_event = conn.event_times.t_event

    @event_times [t_init, t_event]
end
```

!!! warn
    
    It is **very important** to define all of the times that appear in a discrete event condition in `@event_times`. Otherwise, the integrator might not know to stop.

To demonstrate the warning above, the following snippet:

```julia
@connection (...)
    t_init = 0.1
    t_event = conn.event_times.t_event

    @event_times [t_init, t_event]

    @discrete_events begin
        (t == 1) => affect
    end
end
```

Since 1 has not been declared as an `event_time`, the integrator will not step at that time point and thus the event will not trigger. The correct `@event_times` declaration should read as:

```julia
@event_times [t_init, t_event, 1]
```

### `@discrete_events`
This macro is used to actually define the conditions and affects associated with the callback. Every line of this block should be in pair syntax, in which the left-hand side is the condition and the right-hand side is the affect on the system.

In the case that there are multiple discrete events, then `discrete_event_condition` will return `true` when any of the conditions hold.

**Conditions**
A condition must be specified as an expression that will return a Boolean, or else the simulation may error. For example, if an event is meant to be triggered at time `t1`, then the condition should be written as `t == t1`.

!!! warning

    For every time that an event should happen, it must be declared using `@event_times`. This is so that a time stop will be added to the integrator and the discrete event will actually trigger.

**Affects**
The affect should be a list of assignments, which give the updates to various states or parameters of the system. For example, if the event should cause the voltage of a destination blox to reset to some value, then one can write `dst.V = dst.V_reset` as the affect.

```julia
@connection function (conn::MyEventConnection)(src::TAN, dst::Matrisome, t)
    w = conn.weight
    t_event = conn.event_times.t_event

    @equations begin
        jcn = w * dst.TAN_spikes
    end

    @event_times t_event

    @discrete_events begin
        (t == t_event) => (dst.TAN_spikes = w * rand(src.rng, Poisson(src.R)))
    end
end
```
