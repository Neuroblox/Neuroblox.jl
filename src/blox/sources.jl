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

mutable struct ImageStimulus{S}
    const namespace
    const odesystem::S
    currect_dot::Int

    function ImageStimulus(; file, name, namespace, dt, t_stimulus, t_pause)
        S = readdlm(file, ',')
        
        sources = [
            SampledData(S[i,:], dt, name=Symbol("$(name)_$(i)")) 
            for i in Base.OneTo(size(S)[1])
        ]
        system = system_from_parts(sources; name)

        new{typeof(system)}(namespace, system, 1)
    end
end
