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

    function TAN(; name, namespace=nothing, κ=100, λ=1)
        @variables t 
        sts = @variables R(t)=κ 
        ps = @parameters κ=κ spikes_window=0.0 jcn=0.0 [input=true]
        eqs = [
                R ~ minimum([κ, κ/(λ*jcn + eps())])
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

    function SNc(; name, namespace=nothing, κ_DA=1, N_time_blocks=5, DA_reward=10, λ_DA=0.33, t_event=90.0)
        @variables t 
        sts = @variables R(t)=κ_DA R_(t)=κ_DA
        ps = @parameters κ=κ_DA λ_DA=λ_DA jcn=0.0 [input=true] jcn_=0.0 
        eqs = [
                R ~ minimum([κ_DA, κ_DA/(λ_DA*jcn + eps())])
                R_ ~ minimum([κ_DA, κ_DA/(λ_DA*jcn_ + eps())])
              ]

        R_cb = [[t_event+3*eps(t_event)] => [jcn_ ~ jcn]]     

        sys = ODESystem(eqs, t, sts, ps; name = name, discrete_events = R_cb)

        new(sys, N_time_blocks, κ_DA, DA_reward, namespace)
    end
end

(b::SNc)(R_DA) = R_DA #b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward

function get_modulator_state(s::SNc)
    sys = get_namespaced_sys(s)
    return sys.R_
end