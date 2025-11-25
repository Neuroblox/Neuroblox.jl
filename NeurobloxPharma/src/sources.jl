struct PulsesInput <: AbstractSimpleStimulus
    name
    namespace
    system
    times_on
    times_off
    pulse_switch
    baseline
    pulse_amp

    function PulsesInput(; name, namespace=nothing, baseline=0.0, pulse_amp=1.0, t_start=[0], pulse_width=100, pulse_switch=ones(length(t_start)))
        @variables u(t) [output=true, description="ext_input"]
        @parameters I=baseline pulse_amp=pulse_amp
        eqs = [u ~ I]

        on = [t_start[1]] => [I~pulse_amp*pulse_switch[1]]
        off = [t_start[1]+pulse_width] => [I~baseline]

        dc = [on,off]

        for i in collect(2:length(t_start))
            on = [t_start[i]] => [I~pulse_amp*pulse_switch[i]]
            off = [t_start[i]+pulse_width] => [I~baseline]
            push!(dc,on)
            push!(dc,off)
        end

        sys = System(eqs, t, [u], [I, pulse_amp]; name=name, discrete_events=dc)

        times_off = t_start .+ pulse_width 
        new(name, namespace, sys, t_start, times_off, pulse_switch, baseline, pulse_amp)
    end
end

function get_sampled_data(t, t_trial::Real, t_stims::AbstractVector, pixel_data::AbstractVector)
    idx = floor(Int, t / t_trial) + 1
    
    return ifelse(
        (t >= first(t_stims[idx])) && (t <= last(t_stims[idx])), 
        pixel_data[idx], 
        0.0
    )
end

@register_symbolic get_sampled_data(t, t_trial::Real, t_stims::AbstractVector, pixel_data::AbstractVector)

mutable struct ImageStimulus <: AbstractStimulus
    const name
    const namespace
    const system
    const IMG # Matrix[pixels X stimuli]
    const stim_parameters
    const category
    const t_stimulus
    const t_pause
    const N_pixels
    const N_stimuli
    current_pixel::Int

    function ImageStimulus(data::DataFrame; name, namespace, t_stimulus, t_pause)
        N_pixels = DataFrames.ncol(data[!, Not(:category)])
        N_stimuli = DataFrames.nrow(data[!, Not(:category)])

        # Append a row of zeros at the end of data so that indexing can work
        # on the final simulation time step when the index will be `nrow(data)+1`.
        d0 = DataFrame(Dict(n => 0 for n in names(data)))
        append!(data, d0)

        S = transpose(Matrix(data[!, Not(:category)]))

        t_trial = t_stimulus + t_pause
        t_stims = [
            ((i-1)*t_trial, (i-1)*t_trial + t_stimulus)
            for i in Base.OneTo(N_stimuli)
        ]
        # Append a dummy stimulation interval at the end
        # so that index is not out of bounds , similar to data above.
        push!(t_stims, (0,0))

        param_name = :u
        @parameters t
        ps = Vector{Num}(undef, N_pixels)
        reset_eqs = Vector{Equation}(undef, N_pixels)
        for i in Base.OneTo(N_pixels)
            s = Symbol(param_name, "_", i)
            ps[i] = only(@parameters $(s) = S[i,1])
            reset_eqs[i] = ps[i] ~ 0.0
        end

        cb_stop_stim = [t_stimulus] => reset_eqs
        sys = ODESystem(Equation[], t, [], ps; name, discrete_events = cb_stop_stim)
        category = data[!, :category]

        ps_namespaced = namespace_parameters(get_namespaced_sys(sys))

        new(name, namespace, sys, S, ps_namespaced, category, t_stimulus, t_pause, N_pixels, N_stimuli, 1)
    end

    function ImageStimulus(file::String; name, namespace, t_stimulus, t_pause)
        @assert last(split(file, '.')) == "csv" "Image file must be a CSV file."
        data = read(file, DataFrame)
        ImageStimulus(data; name, namespace, t_stimulus, t_pause)
    end
end

increment_pixel!(stim::ImageStimulus) = stim.current_pixel = mod(stim.current_pixel, stim.N_pixels) + 1

struct VoltageClampSource <: AbstractSimpleStimulus
    name
    namespace
    system

    function VoltageClampSource(clamp_schedule::Vector{@NamedTuple{t::T1, V::T2}}; name, namespace=nothing) where {T1, T2}
        @variables V(t)=0 [output=true]
        eqs = [D(V) ~ 0]

        cbs = map(clamp_schedule) do cs
            [cs.t] => [V ~ cs.V]
        end

        sys = System(eqs, t, [V], []; name=name, discrete_events=cbs)

        new(name, namespace, sys)
    end
end
