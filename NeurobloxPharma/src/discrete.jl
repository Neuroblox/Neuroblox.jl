"""
    Matrisome(; t_event=180.0)

A model of matrisome, which is part of the [`Striatum`](@ref) blox, with discrete dynamics. At t_event time during simulation it will store the Cortical input to the Striatum and the internal activity of the Striatum, which can be later used to apply a plasticity rule in the corticostriatal connection. 

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.

See also [`Striatum`](@ref)
"""
@blox struct Matrisome(; name, namespace=nothing, t_event=180.0) <: AbstractDiscrete
    @params H=1 TAN_spikes=0.0 jcn_snapshot=0.0 jcn_t_block=0.0 H_snapshot=1 t_event=float(t_event)
    @states # No states!
    @inputs jcn=0.0
    @computed_properties(
        H_learning = H,
        ρ_snapshot = H_snapshot * jcn_snapshot,
        ρ = H*jcn_t_block,
        ρ_= H_snapshot * jcn_snapshot,
        jcn_= jcn_snapshot
    )
    @equations begin
    end
    @discrete_events (t == t_event + sqrt(eps(t_event))) => (H_snapshot=H, jcn_snapshot=jcn_t_block)
    @event_times t_event + √(eps(t_event))
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{Matrisome}) = false

"""
    Striosome()

A model of striosome, which is part of the [`Striatum`](@ref) blox, with discrete dynamics. It calculates a discretized estimate of the internal activity of a [`Striatum`](@ref) component, which can be used in learning and decision-making tasks, e.g. by comparing the activities of multiple striatal modules and choosing the highest one as the task action.

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.

See also [`Striatum`](@ref)
"""
@blox struct Striosome(; name, namespace=nothing) <: AbstractDiscrete
    @params H=1 jcn_t_block=0.0
    @states
    @computed_properties(
        H_learning = H,
        ρ = H * jcn_t_block,
    )
    @inputs jcn=0.0
    @equations begin
    end
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{Striosome}) = false

"""
    TAN(; κ=100, λ=1, rng=Xoshiro(rand(Int)))

A model of TAN (Tonically Active Neurons), which is part of the Striatum, with discrete dynamics that projects to each striatal module of a model to add a random activity value to striatal [`Matrisome`](@ref). 

Arguments : 
- `κ` : Baseline activity of TAN neurons. 
- `λ` : Gain term on the cumulative activity of the source populations that project to TAN. Typically these are [`Striosome`](@ref) components of each [`Striatum`](@ref) module in a model.

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.

See also [`Striatum`](@ref), [`Striosome`](@ref) and [`Matrisome`](@ref).
"""
@blox struct TAN(; name, namespace=nothing, κ=100, λ=1, rng=Xoshiro(rand(Int))) <: AbstractDiscrete
    @params κ λ jcn_t_block=0.0 rng
    @states
    @computed_properties(
        R = min(κ, κ/(λ*jcn_t_block + √(eps())))
    )
    @inputs jcn=0.0
    @equations begin
    end
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{TAN}) = false


"""
    SNc(; κ_DA=1, DA_reward=10, λ_DA=0.33, t_event=90.0)
   
A model of Substantia Nigra core (SNc) with discrete dynamics. It represents the average activity of neurons within SNc and releases dopamine in an activity-based manner which in turn can modulate the plasticity in connection with [`HebbianModulationPlasticity`](@ref).

Arguments : 
- `κ_DA` : Baseline activity of dopaminergic neurons.
- `DA_reward` : Amount of dopamine released when a reward is received.
- `λ_DA` : Gain term on the cumulative activity of the source populations that project to SNc.
- `t_event` : Timepoint during simulation at which the cumulative input to SNc is stored in order to then be used to modulate plasticity in [`HebbianModulationPlasticity`](@ref) connections.

References: 
1. Pathak, A., Brincat, S.L., Organtzidis, H. et al. Biomimetic model of corticostriatal micro-assemblies discovers a neural code. Nat Commun 2025.
"""
@blox struct SNc(; name, namespace=nothing, κ_DA=1, N_time_blocks=5, DA_reward=10, λ_DA=0.33, t_event=90.0) <: AbstractModulator
    @params(
        κ = κ_DA,
        λ = λ_DA,
        jcn_snapshot=0.0,
        jcn_t_block=0.0,
        t_event
    )
    @states
    @inputs jcn=0.0
    @computed_properties(
        R_snapshot = min(κ, κ/(λ*jcn_snapshot + sqrt(eps()))),
        R = min(κ, κ/(λ*jcn_t_block + sqrt(eps()))),
        R_ = min(κ, κ/(λ*jcn_snapshot + sqrt(eps()))),
    )
    @equations begin
    end
    @discrete_events t == t_event => (jcn_snapshot = jcn_t_block,)
    @event_times t_event
    @extra_fields κ_DA=κ_DA DA_reward=DA_reward N_time_blocks=N_time_blocks
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{SNc}) = false
(b::SNc)(R_DA) = R_DA #b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward
function get_modulator_state(s::SNc)
    return s.R_
end
