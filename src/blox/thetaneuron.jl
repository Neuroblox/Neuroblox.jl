@parameters t
D = Differential(t)

"""
thetaneuron has the following parameters:
    η:     Constant drive
    α_inv: Time to peak of spike
    k:     All-to-all coupling strength
and the following variables:
    θ(t):   Theta neuron state
    g(t):   Synaptic current
    jcn(t): Synaptic input
and returns:
    an ODE System
"""
function thetaneuron(;name, η=η, α_inv=α_inv, k=k)

    params = @parameters η=η α_inv=α_inv k=k
    sts    = @variables θ(t)=0.0 g(t)=0.0 jcn(t)=0.0
    
    eqs = [D(θ) ~ 1-cos(θ) + (1+cos(θ))*(η + k*g),
          D(g) ~ α_inv*(jcn - g)]

    return ODESystem(eqs, t, sts, params; name=name)

end