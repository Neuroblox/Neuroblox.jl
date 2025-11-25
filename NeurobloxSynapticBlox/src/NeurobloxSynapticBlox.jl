module NeurobloxSynapticBlox

using Reexport
@reexport using NeurobloxBase
@reexport using NeurobloxBasics

using NeurobloxBase:
    NeurobloxBase,
    t,
    D,
    namespaced_name,
    namespaced_nameof,
    get_namespaced_sys,
    NeurobloxBase,
    AbstractNeuron,
    AbstractExciNeuron,
    AbstractInhNeuron,
    AbstractReceptor,
    AbstractComposite,
    Connector,
    NoLearningRule,
    get_weight,
    get_density,
    to_metadigraph,
    connectors_from_graph,
    system_from_graph,
    system_from_parts,
    get_connection_matrix,
    connect_action_selection!,
    AbstractActionSelection

using NeurobloxBase.GraphDynamicsInterop:
    recursive_getdefault

using NeurobloxPharma:
    NeurobloxPharma,
    NGNMM_theta,
    Matrisome,
    Striosome,
    TAN,
    SNc,
    ImageStimulus,
    subcortical_connection_matrix,
    run_experiment!,
    ClassificationEnvironment,
    Agent,
    GreedyPolicy,
    HebbianPlasticity,
    HebbianModulationPlasticity,
    get_matrisome,
    get_striosome,
    ProgressMeterMonitor

using ModelingToolkit:
    @parameters,
    @variables,
    System,
    ODESystem,
    Num,
    @named

using GraphDynamics:
    GraphDynamics,
    NotConnected,
    ConnectionRule,
    ConnectionMatrices,
    ConnectionMatrix,
    PartitionedGraphSystem,
    GraphSystem,
    initialize_input,
    subsystem_differential,
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    isstochastic,
    event_times,
    get_tag,
    get_states,
    get_params,
    calculate_inputs,
    partitioned,
    maybe_sparse_enumerate_col,
    calculate_inputs,
    to_subsystem,
    nodes,
    add_connection!,
    connections,
    has_connection,
    system_wiring_rule!,
    make_connection_matrices

using Accessors:
    Accessors,
    @set,
    @reset

using Distributions:
    Distributions,
    Bernoulli,
    Uniform,
    Poisson

using StatsBase:
    StatsBase,
    sample

export
    HHExci,
    HHInhi,
    
    GABA_A_Synapse,
    Glu_AMPA_Synapse,
    Glu_AMPA_STA_Synapse,
    
    NGNMM_theta,

    Matrisome,
    Striosome,
    TAN,
    SNc,

    WinnerTakeAll,
    Cortical,
    Striatum,
    GPi,
    GPe,
    Thalamus,
    STN,

    ImageStimulus,
    
    run_experiment!,
    ProgressMeterMonitor,
    ClassificationEnvironment,
    Agent,
    GreedyPolicy,
    HebbianPlasticity,
    HebbianModulationPlasticity,

    @named

include("neurons.jl")
include("synapses.jl")
include("composite.jl")
include("connections.jl")

get_param_vals(x) = getfield(x, :param_vals)

get_param_syms(x) = getfield(x, :param_syms)
get_state_syms(x) = getfield(x, :state_syms)

get_name(x) = getfield(x, :name)

get_system(x) = getfield(x, :system)

get_system!(x) = let sys = get_system(x)
    if isnothing(sys)
        setfield!(x, :system, make_system(x))
    else
        sys
    end
end
get_param_syms!(x) = let ps = get_param_syms(x)
    if isnothing(ps)
        setfield!(x, :param_syms, make_symbolic_params(x))
    else
        ps
    end
end
get_state_syms!(x) = let sts = get_state_syms(x)
    if isnothing(sts)
        setfield!(x, :state_syms, make_symbolic_states(x))
    else
        sts
    end
end

function set_system!(blox)
    blox.system = make_system(blox)
end

function flattened_getproperty(n, s::Symbol)
    if s == :system
        sys = get_system(n)
        if isnothing(sys)
            set_system!(n)
        else
            sys
        end
    elseif s ∈ fieldnames(typeof(n)) && s !== :name
        getfield(n, s)
    else
        getproperty(get_namespaced_sys(n), s)
    end
end

function Base.getproperty(n::Union{HHExci, HHInhi, GABA_A_Synapse, Glu_AMPA_Synapse, Glu_AMPA_STA_Synapse}, s::Symbol)
    flattened_getproperty(n, s)
end

# Take a NamedTuple of parameter values and turn them into MTK parameters with default values
make_symbolic_params(x) = make_symbolic_params(get_param_vals(x))
@generated function make_symbolic_params(nt::NamedTuple{names}) where {names}
    quote
        @parameters( $((:($(names[i]) = nt[$i]) for i ∈ eachindex(names))...) )
        (; $(names...),)
    end
end

include("GraphDynamicsInterop/GraphDynamicsInterop.jl")


end # module NeurobloxSynapticBlox
