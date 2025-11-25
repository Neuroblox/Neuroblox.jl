get_wtas(n::WinnerTakeAll) = [n]
get_wtas(n) = WinnerTakeAll[]

function get_wtas(b::AbstractComposite)
    mapreduce(x -> get_wtas(x), vcat, b.parts)
end

get_ff_inh_neurons(n::AbstractInhNeuron) = [n]
get_ff_inh_neurons(n) = AbstractInhNeuron[]

function get_ff_inh_neurons(b::Cortical)
    mapreduce(x -> get_ff_inh_neurons(x), vcat, b.parts)
end

function get_ff_inh_num(kwargs, name_blox1)
    get(kwargs, :ff_inh_num) do 
        error("feedforward inhibition neuron number from $name_blox1 is not specified")
    end
end

function get_sta(kwargs, name_blox1, name_blox2)
    get(kwargs, :sta, false)
end
