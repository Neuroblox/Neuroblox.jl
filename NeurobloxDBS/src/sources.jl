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
@blox struct DBS(;
    name,
    namespace=nothing,
    frequency=130.0,
    amplitude=2.5,
    pulse_width=0.066,
    offset=0.0,
    start_time=0.0,
    smooth=1e-4) <: AbstractSimpleStimulus

    @params 
    @states
    @inputs
    @outputs
    @extra_fields stimulus = SquareStimulus(frequency, amplitude, offset, start_time, pulse_width, smooth)
    @equations begin
    end
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
@blox struct ProtocolDBS(;
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
    inter_burst_time=200.0) <: AbstractSimpleStimulus
    
    @params 
    @states
    @inputs
    @outputs
    @extra_fields stimulus = BurstStimulus(frequency, amplitude, offset, start_time, pulse_width, smooth,
                    pulses_per_burst, bursts_per_block,
                    pre_block_time, inter_burst_time)
    @equations begin
    end
end


# ------------------------------------------------------------------------------------------
#  Stimulus functions
# ------------------------------------------------------------------------------------------

abstract type AbstractStimulusFunction <: Function end

# Simple square pulse (constant amplitude, constant duty‑cycle)
struct SquareStimulus{T} <: AbstractStimulusFunction
    frequency_khz :: T   # kHz
    amplitude     :: T
    offset        :: T
    start_time    :: T   # ms
    pulse_width   :: T   # ms
    smooth        :: T
    period        :: T   # ms

    function SquareStimulus(frequency_hz, amplitude, offset, start_time, pulse_width, smooth)

        @assert frequency_hz  ≥ 0 "`frequency` must be non-negative"
        @assert amplitude     ≥ 0 "`amplitude` must be non-negative"
        @assert pulse_width   ≥ 0 "`pulse_width` must be non-negative"
        @assert smooth        ≥ 0 "`smooth` must be non-negative"

        f_khz = frequency_hz / 1000
        period = 1 / f_khz

        if pulse_width ≥ period
            @warn "`pulse_width` ≥ `period`: pulses may overlap (pulse_width = $(pulse_width) ms, period = $(period) ms)"
        end

        Tprom = typeof(promote(frequency_hz, amplitude, offset, start_time, pulse_width, smooth)[1])
        return new{Tprom}(f_khz, amplitude, offset, start_time, pulse_width, smooth, period)
    end
end

# for making Accessors.jl @set work
SquareStimulus(f_khz, amp, off, t0, pw, sm, per) = SquareStimulus(f_khz*1000, amp, off, t0, pw, sm)

function (s::SquareStimulus)(t)
    (;frequency_khz, amplitude, offset, start_time, pulse_width, smooth, period) = s
    ifelse(iszero(smooth),
        square(t, frequency_khz, amplitude, offset, start_time, pulse_width),
        square(t, frequency_khz, amplitude, offset, start_time, pulse_width, smooth)
    )
end

# Burst DBS stimulus
struct BurstStimulus{T} <: AbstractStimulusFunction
    frequency_khz    :: T   # kHz
    amplitude        :: T
    offset           :: T
    start_time       :: T   # ms
    pulse_width      :: T   # ms
    smooth           :: T
    pulses_per_burst :: T
    bursts_per_block :: T
    pre_block_time   :: T   # ms
    inter_burst_time :: T   # ms
    period           :: T   # ms
    burst_duration   :: T   # ms
    cycle_duration   :: T   # ms
    inter_pulse_time :: T   # ms

    function BurstStimulus(frequency_hz, amplitude, offset, start_time,
                           pulse_width, smooth, pulses_per_burst,
                           bursts_per_block, pre_block_time, inter_burst_time)

        @assert frequency_hz        ≥ 0 "`frequency` must be non-negative"
        @assert amplitude           ≥ 0 "`amplitude` must be non-negative"
        @assert pulse_width         ≥ 0 "`pulse_width` must be non-negative"
        @assert smooth              ≥ 0 "`smooth` must be non-negative"
        @assert inter_burst_time    ≥ 0 "`inter_burst_time` must be non-negative"
        @assert isinteger(pulses_per_burst) && pulses_per_burst ≥ 0 "`pulses_per_burst` must be a non-negative integer"
        @assert isinteger(bursts_per_block) && bursts_per_block ≥ 0 "`bursts_per_block` must be a non-negative integer"

        f_khz = frequency_hz / 1000
        period = 1 / f_khz

        if pulse_width ≥ period
            @warn "`pulse_width` ≥ `period`: pulses may overlap (pulse_width = $(pulse_width) ms, period = $(period) ms)"
        end

        # Pre-compute some timing parameters
        burst_duration  = pulses_per_burst * period
        cycle_duration  = burst_duration + inter_burst_time
        inter_pulse_time = max(0, period - pulse_width)
        T = typeof(promote(f_khz, amplitude, offset, start_time, pulse_width, smooth,
                           pulses_per_burst, bursts_per_block,
                           pre_block_time, inter_burst_time)[1])

        return new{T}(f_khz, amplitude, offset, start_time, pulse_width, smooth,
                           pulses_per_burst, bursts_per_block,
                           pre_block_time, inter_burst_time,
                           period, burst_duration, cycle_duration, inter_pulse_time)
    end
end

# for making Accessors.jl @set work
BurstStimulus(f_khz, amp, off, t0, pw, sm, ppb, bpb, pbt, ibt, per, bdu, cd, ipt) =
    BurstStimulus(f_khz*1000, amp, off, t0, pw, sm, ppb, bpb, pbt, ibt)

function (s::BurstStimulus)(t)
    (;frequency_khz, amplitude, offset,
        start_time, pulse_width, smooth,
        pulses_per_burst, bursts_per_block,
        pre_block_time, inter_burst_time, period,
        burst_duration, cycle_duration, inter_pulse_time) = s

    t_adj = t - pre_block_time
    current_burst = floor(t_adj / cycle_duration)
    t_within = t_adj - current_burst * cycle_duration 

    # Nested ifelse structure (for compatibility with Symbolics) determines output at time t:
    # 1. Before protocol starts: return offset
    # 2. After all bursts complete: return offset
    # 3. Between bursts: return offset
    # 4. During burst: return square wave pulse
    ifelse(t_adj < 0,
        offset,
        ifelse(current_burst ≥ bursts_per_block,
            offset,
            ifelse(t_within ≥ burst_duration - inter_pulse_time,
                offset,
                ifelse(iszero(smooth),
                    square(t_within, frequency_khz, amplitude, offset, start_time, pulse_width),
                    square(t_within, frequency_khz, amplitude, offset, start_time, pulse_width, smooth)
                )
            )
        )
    )
end

sawtooth(t,f,off) = f*(t-off) - floor(f*(t-off))

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
@inline function square(t, f, amplitude, offset, start_time, pulse_width)
    ifelse(phase(t, f, start_time) < pulse_width*f,
            offset + amplitude,
            offset)
end

# phase in [0,1)
phase(t, f, t0) = f*(t - t0) - floor(f*(t - t0))

# ------------------------------------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------------------------------------

get_stimulus_function(dbs::Union{DBS, ProtocolDBS}) = dbs.stimulus
function get_stimulus_function(sys::GraphSystem)
    dbs = filter(b -> b isa Union{DBS, ProtocolDBS}, collect(nodes(sys.flat_graph)))
    @assert length(dbs) == 1 "A system can only have one DBS input."
    get_stimulus_function(only(dbs))
end
get_stimulus_function(prob::ODEProblem) = prob.ps[:stimulus_DBSConnection_g₊dbs_n1] #TODO: it's very weird that this is hardcoded this way

# total duration (ms) of a DBS protocol --------------------------------------
function get_protocol_duration(dbs::Union{ProtocolDBS, GraphSystem, ODEProblem})
    stim = get_stimulus_function(dbs)
    @assert !(stim isa SquareStimulus) "You are using an endless SquareStimulus."
    stim.pre_block_time + stim.bursts_per_block * stim.cycle_duration - stim.inter_burst_time
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

function compute_transition_times(stimulus::AbstractStimulusFunction, f , dt, tspan, start_time, pulse_width; atol=0)
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
