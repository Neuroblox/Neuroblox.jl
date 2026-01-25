"""
    PulsesInput(; baseline=0, 
                pulse_amp=1,
                t_start=[0], 
                pulse_width=100, 
    )

A source that applies square wave pulses as input currents to neuron blox. 

Arguments : 
- baseline : Baseline current value that applies when the pulse is off.
- pulse_amp : [μA] Current pulse amplitude.
- t_start : [ms] Vector of values that denote the begin of each pulse in a sequence. there will be as many pulses as elements in t_start.
- pulse_width : [ms] Duration of each pulse in a sequence.
"""
@blox struct PulsesInput(;name, namespace, baseline=0.0, pulse_amp=1.0,
                         t_start=[0], pulse_width=100) <: AbstractSimpleStimulus
    @params(
        times_on  = t_start,
        times_off = t_start .+ pulse_width,
        baseline=float(baseline),
        pulse_amp=float(pulse_amp)
    )
    @states I=float(baseline)
    @inputs
    @outputs I
    @equations begin
        # no evolution via differential equation. I changes via events
        D(I) = 0.0
    end
    @event_times [times_on; times_off]
end
GraphDynamics.has_discrete_events(::Type{PulsesInput}) = true
GraphDynamics.discrete_event_condition(s::Subsystem{PulsesInput}, t, _) = (t ∈ s.times_on || t ∈ s.times_off)
function GraphDynamics.apply_discrete_event!(integrator, sys_view, s::Subsystem{PulsesInput}, _)
    (;t) = integrator
    (;times_on, times_off, pulse_amp, baseline) = s
    for i ∈ eachindex(times_on)
        if t == times_on[i]
            sys_view.I[] = pulse_amp
        end
    end
    if t ∈ times_off
        sys_view.I[] = baseline
    end
    nothing
end

"""
    ImageStimulus(data::DataFrame; t_stimulus, t_pause)

A blox to emulate the presentation of images during a behavioral task.

Arguments : 
- data : A DataFrame object where each row is a separate image and each column is a pixel value of the corresponding image. The images are flattened from a Matrix to a Vector to fit this format.
- t_stimulus : [ms] Timepoint at which an image is presented. This is a single value and applied to all images in data.
- t_pause : [ms] Timepoint at which an image disappears. This emulates pauses that are often added in behavioral tasks after an image is presented and before the participant makes a choice and/or the next trial begins.
"""
@blox struct ImageStimulus(data::DataFrame; name, namespace=nothing,
                           t_stimulus, t_pause) <: AbstractSimpleStimulus
    N_pixels = DataFrames.ncol(data[!, Not(:category)])
    N_stimuli = DataFrames.nrow(data[!, Not(:category)])

    # Append a row of zeros at the end of data so that indexing can work
    # on the final simulation time step when the index will be `nrow(data)+1`.
    d0 = DataFrame(Dict(n => 0 for n in names(data)))
    append!(data, d0)

    IMG = transpose(Matrix(data[!, Not(:category)]))

    t_trial = t_stimulus + t_pause
    t_stims = [((i-1)*t_trial, (i-1)*t_trial + t_stimulus)  for i in 1:N_stimuli]
    # Append a dummy stimulation interval at the end
    # so that index is not out of bounds , similar to data above.
    push!(t_stims, (0,0))
    @params(
        current_image=IMG[:,1],
        IMG,
        category = data[!, :category],
        t_stimulus,
        t_pause,
        N_pixels,
        N_stimuli
    )
    @states
    @inputs
    @equations begin
    end
    @event_times t_stimulus
    @extra_fields current_pixel::Base.RefValue{Int} = Ref(1)
end
GraphDynamics.has_discrete_events(::Type{ImageStimulus}) = true
GraphDynamics.discrete_event_condition(s::Subsystem{ImageStimulus}, t, _) = s.t_stimulus == t
function GraphDynamics.apply_discrete_event!(integrator, sys_view, s::Subsystem{ImageStimulus}, _)
    # zero out the current image
    s.current_image .= 0.0
    nothing
end

increment_pixel!(stim::ImageStimulus) = stim.current_pixel[] = mod(stim.current_pixel[], stim.param_vals.N_pixels) + 1

function ImageStimulus(file::String; name, namespace, t_stimulus, t_pause)
    @assert last(split(file, '.')) == "csv" "Image file must be a CSV file."
    data = read(file, DataFrame)
    ImageStimulus(data; name, namespace, t_stimulus, t_pause)
end


"""
    VoltageClampSource(clamp_schedule::Vector{<:NamedTuple{(:t, :V)}})

A blox to emulate a voltage clamp experiment.

Arguments : 
- clamp_schedule : A vector where each element is a Tuple of the form (t = time_value_A, V = voltage_value_A), representing a schedule for the clamping experiment. Each value corresponds to setting the voltage V of the target neuron to a value voltage_value_A at the timepoint time_value_A.
"""
@blox struct VoltageClampSource(clamp_schedule::Vector{<:NamedTuple{(:t, :V)}}; name, namespace=nothing) <: AbstractSimpleStimulus
    @params(
        clamp_times=map(x -> x.t, clamp_schedule),
        clamp_volages=map(x -> x.V, clamp_schedule),
        V=0.0,
    )
    @states
    @inputs
    @equations begin
    end
    @event_times clamp_times
end
GraphDynamics.has_discrete_events(::Type{VoltageClampSource}) = true
GraphDynamics.discrete_event_condition(s::Subsystem{VoltageClampSource}, t, _) = t ∈ s.clamp_times
function GraphDynamics.apply_discrete_event!(integrator, sys_view, s::Subsystem{VoltageClampSource}, _)
    t = integrator.t
    for (ct, V) ∈ zip(s.clamp_times, s.clamp_volages)
        if t == ct
            sys_view.V[] = V
        end
    end
    nothing
end
