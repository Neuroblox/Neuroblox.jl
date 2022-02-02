@parameters t
D = Differential(t)

function ThetaNeuron(;name, η=η, α_inv=α_inv, k=k)

    params = @parameters η=η α_inv=α_inv k=k
    # η:     Constant drive
    # α_inv: Time to peak of spike
    # k:     All-to-all coupling strength

    sts    = @variables θ(t)=0.0 g(t)=0.0 jcn(t)=0.0
    # θ(t):   Theta neuron state
    # g(t):   Synaptic current
    # jcn(t): Synaptic input

   eqs = [D(θ) ~ 1-cos(θ) + (1+cos(θ))*(η + k*g),
          D(g) ~ α_inv*(jcn - g)]

    return ODESystem(eqs, t, sts, params; name=name)

end