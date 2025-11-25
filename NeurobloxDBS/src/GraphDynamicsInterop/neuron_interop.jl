##----------------------------------------------
## Neurons / Neural Mass
##----------------------------------------------

for sys ∈ [HHNeuronInhib_MSN_Adam(name=:hhni_msn_adam)
           HHNeuronInhib_FSI_Adam(name=:hhni_fsi_adam)
           HHNeuronExci_STN_Adam(name=:hhne_stn_adam)
           HHNeuronInhib_GPe_Adam(name=:hhni_GPe_adam)]
    define_neuron(sys; mod=@__MODULE__)
end