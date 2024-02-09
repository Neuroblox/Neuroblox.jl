abstract type AbstractDiscrete <: AbstractBlox end

abstract type AbstractModulator <: AbstractDiscrete end

struct Matrisome <: AbstractDiscrete
    odesystem
    namespace

    function Matrisome(; name, namespace=nothing, t_event=180.0)
        #HACK : this t_event has to be informed from the t_event in Action Selection block
        @variables t 
        sts = @variables ρ(t)=0.0 ρ_(t)
        #HACK : jcn_ and H_ store the value of jcn and H at time t_event that can be accessed after the simulation
        ps = @parameters H=1 TAN_spikes=0.0 jcn=0.0 [input=true] jcn_=0.0 H_=1 
        eqs = [
            ρ ~ H*jcn,
            ρ_ ~ H_*jcn_
        ]
        cb_eqs = [ jcn_ ~ jcn,
                    H_ ~ H
                 ]
        Rho_cb = [[t_event+3*eps(t_event)] => cb_eqs]   
        sys = ODESystem(eqs, t, sts, ps; name = name, discrete_events = Rho_cb)

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
                ρ ~ H*jcn,
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
        ps = @parameters κ=κ jcn=0.0 [input=true]
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
        ps = @parameters κ=κ_DA λ_DA=λ_DA jcn=0.0 [input=true] jcn_=0.0 #HACK: jcn_ stores the value of jcn at time t_event that can be accessed after the simulation
        eqs = [
                R ~ minimum([κ_DA, κ_DA/(λ_DA*jcn + eps())]),
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