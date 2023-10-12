@parameters t

#CosineSource
mutable struct CosineSource	
    f::Num
    a::Num
    phi::Num
    offset::Num
    tstart::Num
    connector::Num
    odesystem::ODESystem
    function CosineSource(;name, f=18, a=10, phi=0, offset=0, tstart=0)
        @named source = Blocks.Cosine(frequency=f, amplitude=a, phase=phi, 		 
                        offset=offset, start_time=tstart, smooth=false)	
        new(f, a, phi, offset, tstart, source.output.u, source)
    end
end

#CosineBlox
mutable struct CosineBlox
    amplitude::Num
    frequency::Num
    phase::Num
    connector::Num
    odesystem::ODESystem
    function CosineBlox(;name, amplitude=1, frequency=20, phase=0)

        sts    = @variables jcn(t)=0.0 u(t)=0.0
        params = @parameters amplitude=amplitude frequency=frequency phase=phase

        eqs = [u ~ amplitude * cos(2 * pi * frequency * (t) + phase)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, phase, odesys.u, odesys)
    end
end

#NoisyCosineBlox
mutable struct NoisyCosineBlox
    amplitude::Num
    frequency::Num
    connector::Num
    odesystem::ODESystem
    function NoisyCosineBlox(;name, amplitude=1, frequency=20) 

        sts    = @variables  u(t)=0.0 jcn(t)=0.0
        params = @parameters amplitude=amplitude frequency=frequency

        eqs    = [u   ~ amplitude * cos(2 * pi * frequency * (t) + jcn)]
        odesys = ODESystem(eqs, t, sts, params; name=name)

        new(amplitude, frequency, odesys.u, odesys)
    end
end

#PhaseBlox
mutable struct PhaseBlox
    connector::Num
    odesystem::ODESystem
    function PhaseBlox(;name, phase_range=0, phase_data=0) 

        data        = convert(Vector{Float64}, phase_data)
        range       = convert(Vector{Float64}, phase_range)
        phase_input = CubicSpline(data, range)

        sts         = @variables  u(t)=0.0 jcn(t)=0.0

        eqs         = [u ~ phase_input(t)]
        odesys      = ODESystem(eqs, t, sts, []; name=name)

        new(odesys.u, odesys)
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

mutable struct ImageStimulus <: StimulusBlox
    const namespace
    const odesystem
    const image
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

        S = (transpose ∘ Matrix)(data[!, Not(:category)])

        t_trial = t_stimulus + t_pause
        t_stims = [
            ((i-1)*t_trial, (i-1)*t_trial + t_stimulus)
            for i in Base.OneTo(N_stimuli)
        ]
        # Append a dummy stimulation interval at the end
        # so that index is not out of bounds , similar to data above.
        push!(t_stims, (0,0))

        state_name = :u
        @parameters t
        sts = Vector{Num}(undef, N_pixels)
        eqs = Vector{Equation}(undef, N_pixels)
        for i in Base.OneTo(N_pixels)
            s = Symbol(state_name, "_", i)
            sts[i] = only(@variables $(s)(t) = 0.0) 
            eqs[i] = sts[i] ~ get_sampled_data(t, t_trial, t_stims, S[i,:])
        end

        system = ODESystem(eqs, t, sts, []; name)
        category = data[!, :category]

        new(namespace, system, S, category, t_stimulus, t_pause, N_pixels, N_stimuli, 1)
    end

    function ImageStimulus(file::String; name, namespace, t_stimulus, t_pause)
        @assert last(split(file, '.')) == "csv" "Image file must be a CSV file."
        data = read(file, DataFrame)
        ImageStimulus(data; name, namespace, t_stimulus, t_pause)
    end
end
