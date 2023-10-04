abstract type AbstractDiscrete end

struct DiscreteSpikes <: AbstractDiscrete
    odesystem
    namespace

    function DiscreteSpikes(; name, namespace=nothing)

        @variables t 
        sts = @variables jcn(t)=0.0 [input=true]
        ps = @parameters H=1

        eqs = [
            ρ ~ H*jcn
            jcn ~ 0
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

struct DiscreteInvSpikes <: AbstractDiscrete
    odesystem
    namespace

    function DiscreteInvSpikes(; name, namespace=nothing, κ=0.2)

        @variables t 
        sts = @variables R(t) jcn(t) [input=true]
        ps = @parameters κ=κ
        
        eqs = [
            R ~ IfElse.ifelse(iszero(jcn), κ, κ/jcn)
        ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

abstract type AbstractModulator <: AbstractDiscrete end

struct SNc <: AbstractModulator
    odesystem
    N_time_blocks
    κ_DA
    DA_reward
    namespace

    function SNc(; name, namespace=nothing, κ_DA=0.2, N_time_blocks=5, DA_reward=10)
        @variables t 
        sts = @variables R(t)  jcn(t) [input=true]
        ps = @parameters κ=κ_DA
        eqs = [
            R ~ IfElse.ifelse(iszero(jcn), κ, κ/jcn)
            jcn ~ 0
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