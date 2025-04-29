# Defines a DBS (Deep Brain Stimulation) stimulus that can be either continuous or burst protocol
struct DBS <: StimulusBlox
    params::Vector{Num}
    system::ODESystem
    namespace::Union{Symbol, Nothing}
    stimulus::Function
end

"""
    DBS(; name, namespace=nothing, frequency=130.0, amplitude=2.5, pulse_width=0.066, 
        offset=0.0, start_time=0.0, smooth=1e-4)

Create a continuous deep brain stimulation (DBS) stimulus with regular pulses.

Arguments:
- name: Name given to ODESystem object within the blox
- namespace: Additional namespace above name if needed for inheritance
- frequency: Pulse frequency in Hz 
- amplitude: Pulse amplitude in arbitrary units
- pulse_width: Duration of each pulse in ms
- offset: Baseline value of the signal between pulses
- start_time: Time delay before stimulation begins in ms
- smooth: Smoothing parameter for pulse transitions, set to 0 for sharp transitions

Returns a DBS stimulus blox that outputs square pulses with specified parameters.
"""
function DBS(;
    name,
    namespace=nothing,
    frequency=130.0,
    amplitude=2.5,
    pulse_width=0.066,
    offset=0.0,
    start_time=0.0,
    smooth=1e-4
)
    # Ensure consistent numeric types for all parameters
    frequency, amplitude, pulse_width, offset, start_time, smooth = 
        promote(frequency, amplitude, pulse_width, offset, start_time, smooth)

    smoothing = smooth == 0 ? false : true

    p = paramscoping(
        tunable=true;
        frequency=frequency,
        amplitude=amplitude,
        pulse_width=pulse_width,
        offset=offset,
        start_time=start_time,
        smooth=smooth,
    )

    frequency, amplitude, pulse_width, offset, start_time, smooth = p

    # Convert to kHz (to match interal time in ms)
    frequency_khz = frequency/1000.0

    # Create stimulus function based on smooth/non-smooth square wave
    stimulus = if smoothing == 0
        t -> square(t, frequency_khz, amplitude, offset, start_time, pulse_width)
    else
        t -> square(t, frequency_khz, amplitude, offset, start_time, pulse_width, smooth)
    end

    sts = @variables u(t) [output = true]
    eqs = [u ~ stimulus(t)]
    sys = System(eqs, t, sts, p; name=name)

    # Get the stimulus function with the default parameters ready to use
    eq       = first(equations(sys))
    expr     = eq.rhs
    params   = ModelingToolkit.parameters(sys)
    defs     = ModelingToolkit.getdefault.(params)
    expr_sub = Symbolics.substitute(expr, Dict(params .=> defs))
    f_expr   = build_function(expr_sub, t; expression=Val{false})
    stimulus = eval(f_expr)
    
    DBS(p, sys, namespace, stimulus)
end

"""
    ProtocolDBS(; name, namespace=nothing, frequency=130.0, amplitude=2.5,
                  pulse_width=0.066, offset=0.0, start_time=0.0, smooth=1e-4,
                  pulses_per_burst=10, bursts_per_block=12, 
                  pre_block_time=200.0, inter_burst_time=200.0)

Create a deep brain stimulation (DBS) stimulus consisting of a block of pulse bursts.

Arguments:
- name: Name given to ODESystem object within the blox
- namespace: Additional namespace above name if needed for inheritance
- frequency: Pulse frequency in Hz
- amplitude: Pulse amplitude in arbitrary units  
- pulse_width: Duration of each pulse in ms
- offset: Baseline value of the signal between pulses
- start_time: Time delay before stimulation begins in ms
- smooth: Smoothing parameter for pulse transitions, set to 0 for sharp transitions
- pulses_per_burst: Number of pulses in each burst
- bursts_per_block: Number of bursts in the block
- pre_block_time: Time before the block starts in ms
- inter_burst_time: Time between bursts in ms

Returns a DBS stimulus blox that outputs a block of pulse bursts.
"""
function ProtocolDBS(;
    name,
    namespace=nothing,
    frequency=130.0,
    amplitude=2.5,
    pulse_width=0.066,
    offset=0.0,
    start_time=0.0,
    smooth=1e-4,
    pulses_per_burst=10,
    bursts_per_block=12,
    pre_block_time=200.0,
    inter_burst_time=200.0
)
    # Ensure consistent numeric types for all parameters
    frequency, amplitude, pulse_width, offset, start_time, smooth, pre_block_time, inter_burst_time = 
        promote(frequency, amplitude, pulse_width, offset, start_time, smooth, pre_block_time, inter_burst_time)

    p = paramscoping(
        tunable=true;
        frequency=frequency,
        amplitude=amplitude,
        pulse_width=pulse_width,
        offset=offset,
        start_time=start_time,
        smooth=smooth,
        pulses_per_burst=pulses_per_burst,
        bursts_per_block=bursts_per_block,
        pre_block_time=pre_block_time,
        inter_burst_time=inter_burst_time,
    )

    frequency, amplitude, pulse_width, offset, start_time, smooth, pulses_per_burst, bursts_per_block, pre_block_time, inter_burst_time = p

    # Convert to kHz (to match interal time in ms)
    frequency_khz = frequency/1000.0

    # Pre-compute timing parameters for the protocol
    pulse_period = 1/frequency_khz  
    burst_duration = pulse_period * pulses_per_burst
    burst_plus_gap = burst_duration + inter_burst_time
    
    function protocol_stimulus(t)
        # Compute timing relative to protocol start
        t_adjusted = t - pre_block_time                            # Time since protocol start
        current_burst = floor(t_adjusted / burst_plus_gap)         # Current burst number
        t_within_burst_cycle = t_adjusted - current_burst * burst_plus_gap  # Time within current burst
        
        # Nested ifelse structure (for compatibility with Symbolics) determines output at time t:
        # 1. Before protocol starts: return offset
        # 2. After all bursts complete: return offset
        # 3. Between bursts: return offset
        # 4. During burst: return square wave pulse
        ifelse(t < pre_block_time,
            offset,
            ifelse(current_burst >= bursts_per_block,
                offset,
                ifelse(t_within_burst_cycle >= burst_duration - pulse_width/2,
                    offset,
                    ifelse(smooth == 0,
                        square(t_within_burst_cycle, frequency_khz, amplitude, offset, start_time, pulse_width),
                        square(t_within_burst_cycle, frequency_khz, amplitude, offset, start_time, pulse_width, smooth)
                    )
                )
            )
        )
    end

    sts = @variables u(t) [output = true]
    eqs = [u ~ protocol_stimulus(t)]
    sys = System(eqs, t, sts, p; name=name)
    
    # Get the stimulus function with the default parameters ready to use
    eq       = first(equations(sys))
    expr     = eq.rhs
    params   = ModelingToolkit.parameters(sys)
    defs     = ModelingToolkit.getdefault.(params)
    expr_sub = Symbolics.substitute(expr, Dict(params .=> defs))
    f_expr   = build_function(expr_sub, t; expression=Val{false})
    stimulus = eval(f_expr)

    DBS(p, sys, namespace, stimulus)
end

"""
Get the DBS stimulus function as a function of time and all other parameters of the DBS blox,
in contrast to the stimulus function stored in the blox, which is a function of time only.

If the blox is a simple DBS blox, the returned stimulus function is called as:

```julia
dbs = DBS()
stimulus = get_stimulus_function(dbs)
stimulus(t,
        frequency,
        amplitude,
        pulse_width,
        offset,
        start_time,
        smooth)
```

If the blox is a protocol DBS, the returned stimulus function is called as:

```julia
dbs = ProtocolDBS()
stimulus = get_stimulus_function(dbs)
stimulus(t,
        frequency,
        amplitude,
        pulse_width,
        offset,
        start_time,
        smooth,
        pulses_per_burst,
        bursts_per_block,
        pre_block_time,
        inter_burst_time)
```
"""
function get_stimulus_function(dbs)
    eq       = first(equations(dbs.system))
    expr     = eq.rhs

    time_sym = ModelingToolkit.get_iv(dbs.system)
    params   = ModelingToolkit.parameters(dbs.system)

    f_expr = build_function(
        expr,
        time_sym,
        params...;
        expression=Val{false}
    )
    f = eval(f_expr)
end

function sawtooth(t, f, offset)
    f * (t - offset) - floor(f * (t - offset))
end

# Smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width, δ)
    invδ = 1 / δ
    pulse_width_fraction = pulse_width * f
    threshold = 1 - 2 * pulse_width_fraction
    amp_half = 0.5 * amplitude
    start_time = start_time + 0.5 * pulse_width

    saw = sawtooth(t, f, start_time)
    triangle_wave = 4 * abs(saw - 0.5) - 1
    y = amp_half * (1 + tanh(invδ * (triangle_wave - threshold))) + offset

    return y
end

# Non-smoothed square pulses
function square(t, f, amplitude, offset, start_time, pulse_width)

    saw1 = sawtooth(t - start_time, f, pulse_width)
    saw2 = sawtooth(t - start_time, f, 0)
    saw3 = sawtooth(-start_time, f, pulse_width)
    saw4 = sawtooth(-start_time, f, 0)
    
    y = amplitude * (saw1 - saw2 - saw3 + saw4) + offset

    return y
end

function detect_transitions(t, signal::Vector{T}; atol=0) where T <: AbstractFloat
    low = minimum(signal)
    high = maximum(signal)

    # Get indexes when the signal is approximately equal to its low and high values
    low_inds = isapprox.(signal, low; atol=atol)
    high_inds = isapprox.(signal, high; atol=atol)

    # Detect each type of transitions
    trans_inds_1 = diff(low_inds) .== 1 
    trans_inds_2 = diff(low_inds) .== -1 
    trans_inds_3 = diff(high_inds) .== 1 
    trans_inds_4 = diff(high_inds) .== -1 
    circshift!(trans_inds_1, -1)
    circshift!(trans_inds_3, -1)

    # Combine all transition
    transitions_inds = trans_inds_1 .| trans_inds_2 .| trans_inds_3 .| trans_inds_4
    pushfirst!(transitions_inds, false)

    return transitions_inds
end

function compute_transition_times(stimulus::Function, f , dt, tspan, start_time, pulse_width; atol=0)
    period = 1000.0 / f
    n_periods = floor((tspan[end] - start_time) / period)

    # Detect single pulse transition points
    t = (start_time + 0.5 * period):dt:(start_time + 1.5 * period)
    s = stimulus.(t)
    transitions_inds = detect_transitions(t, s, atol=atol)
    single_pulse = t[transitions_inds]

    # Calculate pulse times across all periods
    period_offsets = (-1:n_periods+1) * period
    pulses = single_pulse .+ period_offsets'
    transition_times = vec(pulses)

    # Filter estimated times within the actual time range
    inds = (transition_times .>= tspan[1]) .& (transition_times .<= tspan[end])
    
    return transition_times[inds]
end

function compute_transition_values(transition_times, t, signal)

    # Ensure transition_points are within the range of t, assuming both are ordered
    @assert begin
        t[1] <= transition_times[1]
        transition_times[end] <= t[end] 
    end "Transition points must be within the range of t"

    # Find the indices of the closest time points
    indices = searchsortedfirst.(Ref(t), transition_times)
    transition_values = signal[indices]
    
    return transition_values
end

function get_protocol_duration(dbs::DBS)

    # Check if this is a protocol DBS by looking at the number of parameters (in the future we may create a DBS subtype)
    if length(dbs.params) < 10
        error("This DBS object does not contain protocol parameters")
    end
    
    # Access parameters in correct order based on paramscoping
    frequency = ModelingToolkit.getdefault(dbs.params[1])
    pulses_per_burst = ModelingToolkit.getdefault(dbs.params[7])
    bursts_per_block = ModelingToolkit.getdefault(dbs.params[8])
    pre_block_time = ModelingToolkit.getdefault(dbs.params[9])
    inter_burst_time = ModelingToolkit.getdefault(dbs.params[10])

    # Calculate total protocol duration
    pulse_period = 1000.0/frequency
    burst_duration = pulses_per_burst * pulse_period
    block_duration = bursts_per_block * (burst_duration + inter_burst_time) - inter_burst_time
    
    return pre_block_time + block_duration
end

function get_inds(x::Union{Vector{String}, Vector{Symbol}}, pattern::Regex)
    findall(x -> contains(string(x), pattern), x)
end

function get_param_value(prob, params, params_str, param_name)
    ind = get_inds(params_str, Regex("₊$(param_name)\$"))
    @assert length(ind) == 1 "More than one parameter '$param_name' found"
    symbol = params[ind]
    getter = getp(prob, symbol)
    param_value = getter(prob)[1]
    return param_value
end

function get_protocol_duration(prob::SciMLBase.AbstractDEProblem)

    params = parameters(prob.f.sys)
    param_str = string.(params)
    frequency = get_param_value(prob, params, param_str, "frequency")
    pulses_per_burst = get_param_value(prob, params, param_str, "pulses_per_burst")
    bursts_per_block = get_param_value(prob, params, param_str, "bursts_per_block")
    pre_block_time = get_param_value(prob, params, param_str, "pre_block_time")
    inter_burst_time = get_param_value(prob, params, param_str, "inter_burst_time")

    # Calculate total protocol duration
    pulse_period = 1000.0/frequency
    burst_duration = pulses_per_burst * pulse_period
    block_duration = bursts_per_block * (burst_duration + inter_burst_time) - inter_burst_time
    
    return pre_block_time + block_duration
end