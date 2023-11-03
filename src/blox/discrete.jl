abstract type AbstractAlgebraic end

abstract type AbstractModulator <: AbstractAlgebraic end

linear_func(jcn, H) = H*jcn
@register_symbolic linear_func(jcn, H)

struct Matrisome <: AbstractAlgebraic
    odesystem
    namespace

    function Matrisome(; name, namespace=nothing)
        @variables t 
        sts = @variables jcn(t)=0.0 [input=true, irreducible=true]
        ps = @parameters H=1
        sys = ODESystem(Equation[], t, sts, ps; name)

        new(sys, namespace)
    end
end

struct Striosome <: AbstractAlgebraic
    odesystem
    namespace

    function Striosome(; name, namespace=nothing)
        @variables t 
        sts = @variables jcn(t)=0.0 [input=true]
        ps = @parameters H=1
        sys = ODESystem(Equation[], t, sts, ps; name)

        new(sys, namespace)
    end
end

cases_func(jcn, κ) = IfElse.ifelse(iszero(jcn), κ, κ/jcn)

struct TAN <: AbstractAlgebraic
    odesystem
    namespace

    function TAN(; name, namespace=nothing, κ=0.2)
        @variables t 
        sts = @variables jcn(t)=0.0 [input=true, irreducible=true]
        ps = @parameters κ=κ spikes_window=0.0
        sys = ODESystem(Equation[], t, sts, ps; name)

        new(sys, namespace)
    end
end

struct SNc <: AbstractModulator
    odesystem
    N_time_blocks
    κ_DA
    DA_reward
    namespace

    function SNc(; name, namespace=nothing, κ_DA=0.2, N_time_blocks=5, DA_reward=10)
        @variables t 
        sts = @variables jcn(t)=0.0 [input=true]
        ps = @parameters κ=κ_DA
        sys = ODESystem(Equation[], t, sts, ps; name)

        new(sys, N_time_blocks, κ_DA, DA_reward, namespace)
    end
end

(b::SNc)(R_DA, feedback) = b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward

function get_modulator_value(s::SNc, sol, t)
    jcn = get_modulator_state(s, sol, t)

    return cases_func(jcn, s.κ_DA)
end

function get_modulator_state(s::SNc, sol, t)
    sys = get_namespaced_sys(s)

    return only(sol(t; idxs = [sys.jcn]))
end