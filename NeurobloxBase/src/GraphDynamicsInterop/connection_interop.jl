##----------------------------------------------
## Connections
##----------------------------------------------

function define_basic_connection(c::Connector, blox_src::TSrc, blox_dst::TDst; mod=@__MODULE__()) where {TSrc, TDst}
    name_src = only(c.source)
    name_dst = only(c.destination)
    if isempty(c.weight)
        w = :_
    else
        w = only(c.weight)
    end 

    sys_src = get_namespaced_sys(blox_src)
    sys_dst = get_namespaced_sys(blox_dst)
    
    state_rules_src = map(ModelingToolkit.namespace_variables(sys_src)) do s
        @rule s => Expr(:., name_src, QuoteNode(Symbol(split(string(s.f), '₊')[end])))
    end
    param_rules_src = map(ModelingToolkit.namespace_parameters(sys_src)) do s
        @rule s => Expr(:., name_src, QuoteNode(Symbol(split(string(s), '₊')[end])))
    end
    state_rules_dst = map(ModelingToolkit.namespace_variables(sys_dst)) do s
        @rule s => Expr(:., name_dst, QuoteNode(Symbol(split(string(s.f), '₊')[end])))
    end
    param_rules_dst = map(ModelingToolkit.namespace_parameters(sys_dst)) do s
        @rule s => Expr(:., name_dst, QuoteNode(Symbol(split(string(s), '₊')[end])))
    end
    r = (Postwalk ∘ Chain)([[@rule w => Symbol(w)];
                            state_rules_src;
                            param_rules_src;
                            state_rules_dst;
                            param_rules_dst])

    nt = initialize_input(to_subsystem(blox_dst))
    length(c.equation) <= length(nt) || error("Too many equations for destination blox")
    eqs = map(keys(nt)) do lhs
        i = findfirst(c.equation) do eq
            Symbol(split(string(eq.lhs.f), "₊")[end]) == lhs
        end
        rhs = if isnothing(i)
            nt[lhs]
        else
            toexpr(r(c.equation[i].rhs))
        end
        Expr(:(=), lhs, rhs)
    end

    @eval mod begin
        function $GraphDynamics.system_wiring_rule!(h, blox_src::$TSrc, blox_dst::$TDst; weight, kwargs...)
            conn = BasicConnection(weight)
            add_connection!(g, blox_src, blox_dst; conn, kwargs..., weight, conn)
        end
        function (c::$BasicConnection)($name_src::$Subsystem{$TSrc}, $name_dst::$Subsystem{$TDst})
            $(Symbol(w)) = c.weight
            $(Expr(:tuple, eqs...))
        end
    end
end

function GraphDynamics.system_wiring_rule!(g, blox_src::AbstractBlox, blox_dst::AbstractBlox; weight, kwargs...)
    # The fallback for all blox: just use BasicConnection
    conn = BasicConnection(weight)
    if blox_src isa AbstractComposite || blox_dst isa AbstractComposite
        name_src = namespaced_nameof(blox_src)
        name_dst = namespaced_nameof(blox_dst)
        error("Tried to connect a composite blox using the fallback wiring rule, but this rule only works for non-composite blox. Source blox: $name_src, Destination blox: $name_dst")
    end
    add_connection!(g, blox_src, blox_dst; conn, weight, kwargs...)
end

function GraphDynamics.system_wiring_rule!(g,
                                           blox_src::Union{AbstractNeuron, AbstractNeuralMass},
                                           blox_dst::Union{AbstractNeuron, AbstractNeuralMass};
                                           weight, connection_rule="basic", kwargs...)
    conn = if connection_rule == "basic"
        BasicConnection(weight)
    elseif connection_rule == "psp"
        PSPConnection(weight)
    else
        ArgumentError("Unrecognized connection rule type, got $(connection_rule), expected either \"basic\" or \"psp\".")
    end
    add_connection!(g, blox_src, blox_dst; conn, weight, kwargs...)
end


##----------------------------------------------

struct BasicConnection{T} <: ConnectionRule
    weight::T
    BasicConnection{T}(x) where {T} = new{T}(x)
    BasicConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::BasicConnection, name_src, name_dst)
    (; weight = Symbol(:w_, name_src, :_, name_dst))
end
Base.zero(::Type{<:BasicConnection{T}}) where {T} = BasicConnection(zero(T))
Base.zero(::Type{BasicConnection}) = BasicConnection(0.0)

function (c::BasicConnection)(blox_src, blox_dst, t)
    (; jcn = c.weight * output(blox_src))
end

##----------------------------------------------

struct EventConnection{T, NT <: NamedTuple} <: ConnectionRule
    weight::T
    event_times::NT
    EventConnection(w::T, event_times::NT) where {T, NT} = new{float(T), NT}(w, event_times)
    EventConnection{T}(w, event_times::NT) where {T, NT} = new{T, NT}(w, event_times)
end
Base.zero(::Type{<:EventConnection{T}}) where {T} = EventConnection(zero(T), (;))
Base.zero(::Type{<:EventConnection}) = EventConnection(0.0, (;))

GraphDynamics.has_discrete_events(::Type{<:EventConnection}, ::Type, ::Type) = true
function GraphDynamics.discrete_event_condition((;event_times)::EventConnection, t, sys_src, sys_dst)
    t ∈ event_times
end
GraphDynamics.event_times((;event_times)::EventConnection, sys_src, sys_dst) = event_times


##----------------------------------------------

struct ReverseConnection{T} <: ConnectionRule
    weight::T
    ReverseConnection{T}(x) where {T} = new{T}(x)
    ReverseConnection(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{ReverseConnection{T}}) where {T} = ReverseConnection(zero(T))
Base.zero(::Type{ReverseConnection}) = ReverseConnection(0.0)

##----------------------------------------------

struct PSPConnection{T} <: ConnectionRule
    weight::T
    PSPConnection{T}(x) where {T} = new{T}(x)
    PSPConnection(x::T) where {T} = new{float(T)}(x)
end
function GraphDynamics.connection_property_namemap(::PSPConnection, name_src, name_dst)
    (; weight = Symbol(:w_PSP_, name_src, :_, name_dst))
end
Base.zero(::Type{PSPConnection}) = PSPConnection(0.0)
Base.zero(::Type{PSPConnection{T}}) where {T} = PSPConnection(zero(T))

function (c::PSPConnection)(sys_src::Subsystem{<:AbstractNeuron}, sys_dst::Subsystem{<:AbstractNeuron}, t)
    (;jcn = c.weight * sys_src.G * (sys_src.E_syn - sys_dst.V))
end

##----------------------------------------------

##----------------------------------------------

struct HHConnection_GAP{T} <: ConnectionRule
    w_gap::T
    HHConnection_GAP{T}(x) where {T} = new{T}(x)
    HHConnection_GAP(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP}) = HHConnection_GAP(0.0)
Base.zero(::Type{HHConnection_GAP{T}}) where {T} = HHConnection_GAP(zero(T))

function GraphDynamics.connection_property_namemap(::HHConnection_GAP, name_src, name_dst)
    (; w_gap = Symbol(:w_GAP_, name_src, :_, name_dst))
end

##----------------------------------------------

struct HHConnection_GAP_Reverse{T} <: ConnectionRule
    w_gap_rev::T
    HHConnection_GAP_Reverse{T}(x) where {T} = new{T}(x)
    HHConnection_GAP_Reverse(x::T) where {T} = new{float(T)}(x)
end
Base.zero(::Type{HHConnection_GAP_Reverse{T}}) where {T} = HHConnection_GAP_Reverse(zero(T))
Base.zero(::Type{HHConnection_GAP_Reverse}) = HHConnection_GAP_Reverse(0.0)

function GraphDynamics.connection_property_namemap(::HHConnection_GAP_Reverse, name_src, name_dst)
    (; w_gap_rev = Symbol(:w_GAP_reverse_, name_src, :_, name_dst))
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, ::AbstractActionSelection; kwargs...)
    #@info "Skipping the wiring of an ActionSelection"
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::AbstractBlox, ::AbstractActionSelection; kwargs...)
    # @info "Skipping the wiring of an ActionSelection"
    nothing
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::AbstractNeuralMass, blox_dest::NeurobloxBase.AbstractObserver; weight, kwargs...)
    add_connection!(g, blox_src, blox_dest; conn = BasicConnection(weight), weight, kwargs...)
end
