for sys ∈ [WilsonCowan(name=:wc)
           HarmonicOscillator(name=:ho)
           JansenRit(name=:jr)  # Note! Regular JansenRit can support delays, and I have not yet implemented this!
           IFNeuron(name=:if)
           LIFNeuron(name=:lif) 
           QIFNeuron(name=:qif)
           IzhikevichNeuron(name=:izh)
           PINGNeuronExci(name=:pexci)
           PINGNeuronInhib(name=:pinhib)
           LinearNeuralMass(name=:lnm)
           OUProcess(name=:ou)
           VanDerPol{NonNoisy}(name=:VdP)
           VanDerPol{Noisy}(name=:VdPN)
           KuramotoOscillator{NonNoisy}(name=:ko)
           KuramotoOscillator{Noisy}(name=:kon)]
    define_neuron(sys; mod=@__MODULE__())
end

for sys ∈ [LIFExciNeuron(name=:lif_exci)
           LIFInhNeuron(name=:lif_inh)]
    define_neuron(sys; mod=@__MODULE__(), generate_discrete_event_functions=false)
end
