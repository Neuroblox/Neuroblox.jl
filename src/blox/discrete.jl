abstract type AbstractDiscrete <: AbstractBlox end

abstract type AbstractModulator <: AbstractDiscrete end

struct Matrisome <: AbstractDiscrete
    odesystem
    namespace

    function Matrisome(; name, namespace=nothing)
        @variables t 
        sts = @variables ρ(t)=0.0 
        ps = @parameters H=1 jcn=0.0 [input=true]
        eqs = [
            ρ ~ H*jcn
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

struct Striosome <: AbstractDiscrete
    odesystem
    namespace

    function Striosome(; name, namespace=nothing)
        @variables t 
        sts = @variables ρ(t)=0.0 
        ps = @parameters H=1 jcn=0.0 [input=true]
        eqs = [
            ρ ~ H*jcn
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

struct TAN <: AbstractDiscrete
    odesystem
    namespace

    function TAN(; name, namespace=nothing, κ=0.2)
        @variables t 
        sts = @variables R(t)=κ 
        ps = @parameters κ=κ spikes_window=0.0 jcn=0.0 [input=true]
        eqs = [
            R ~ IfElse.ifelse(iszero(jcn), κ, κ/jcn)
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

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
        sts = @variables R(t)=κ_DA 
        ps = @parameters κ=κ_DA jcn=0.0 [input=true]
        eqs = [
            R ~ IfElse.ifelse(iszero(jcn), κ, κ/jcn)
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, N_time_blocks, κ_DA, DA_reward, namespace)
    end
end

(b::SNc)(R_DA, feedback) = b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward

function get_modulator_state(s::SNc)
    sys = get_namespaced_sys(s)
    return sys.R
end