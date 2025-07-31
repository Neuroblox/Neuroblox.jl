abstract type AbstractDiscrete <: AbstractBlox end

abstract type AbstractModulator <: AbstractDiscrete end

struct Matrisome <: AbstractDiscrete
    system
    namespace
    t_event
    function Matrisome(; name, namespace=nothing, t_event=180.0)
        #HACK : this t_event has to be informed from the t_event in Action Selection block
        # HACK: H_learning is a state version of the H parameter. 
        # It will be simplified away, but we need to access its value in the solution object 
        # to calculate the weight_gradient for reinforcement learning after the simulation.
        sts = @variables ρ(t) ρ_(t) H_learning(t)
        #HACK : jcn_ and H_ store the value of jcn and H at time t_event that can be accessed after the simulation
        ps = @parameters H=1 TAN_spikes=0.0 jcn=0 [input=true] jcn_=0.0 H_=1
        eqs = [
            ρ ~ H*jcn,
            ρ_ ~ H_*jcn_,
            H_learning ~ H
        ]
        cb_eqs = [ jcn_ ~ jcn,
                    H_ ~ H
                 ]
        Rho_cb = [[t_event + sqrt(eps(t_event))] => cb_eqs]   
        sys = ODESystem(eqs, t, sts, ps; name = name, discrete_events = Rho_cb)

        new(sys, namespace, t_event)
    end
end

struct Striosome <: AbstractDiscrete
    system
    namespace

    function Striosome(; name, namespace=nothing)
        # HACK: H_learning is a state version of the H parameter. 
        # It will be simplified away, but we need to access its value in the solution object 
        # to calculate the weight_gradient for reinforcement learning after the simulation.
        sts = @variables ρ(t) H_learning(t)
        ps = @parameters H=1 jcn=0 [input=true]
        eqs = [
                ρ ~ H*jcn,
                H_learning ~ H
              ]
       
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

struct TAN <: AbstractDiscrete
    system
    namespace

    function TAN(; name, namespace=nothing, κ=100, λ=1)
        sts = @variables R(t)
        ps = @parameters κ=κ λ=λ jcn=0 [input=true]
        eqs = [
                R ~ min(κ, κ/(λ*jcn + sqrt(eps())))
              ]
        sys = ODESystem(eqs, t, sts, ps; name)

        new(sys, namespace)
    end
end

struct SNc <: AbstractModulator
    system
    N_time_blocks
    κ_DA
    DA_reward
    namespace
    t_event

    function SNc(; name, namespace=nothing, κ_DA=1, N_time_blocks=5, DA_reward=10, λ_DA=0.33, t_event=90.0) 
        sts = @variables R(t) R_(t) 
        ps = @parameters κ=κ_DA λ=λ_DA jcn=0 [input=true] jcn_=0 #HACK: jcn_ stores the value of jcn at time t_event that can be accessed after the simulation

        eqs = [
                R ~ min(κ, κ/(λ*jcn + sqrt(eps()))),
                R_ ~ min(κ, κ/(λ*jcn_ + sqrt(eps())))
              ]

        R_cb = [[t_event] => [jcn_ ~ jcn]]     

        sys = ODESystem(eqs, t, sts, ps; name = name, discrete_events = R_cb)

        new(sys, N_time_blocks, κ_DA, DA_reward, namespace, t_event)
    end
end

(b::SNc)(R_DA) = R_DA #b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward

function get_modulator_state(s::SNc)
    sys = get_namespaced_sys(s)
    return sys.R_
end
