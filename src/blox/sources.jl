@parameters t

#CosineBlox
mutable struct CosineBlox	
    f::Num
    a::Num
    phi::Num
    offset::Num
    tstart::Num
    connector::Num
    odesystem::ODESystem
    function CosineBlox(;name, f=18, a=10, phi=0, offset=0, tstart=0)
        @named source = Cosine(frequency=f, amplitude=a, phase=phi, 		 
                        offset=offset, start_time=tstart, smooth=false)	
        new(f, a, phi, offset, tstart, source.output.u, source)
    end
end
const cosine_source = CosineBlox

#NoisyCosineBlox
mutable struct NoisyCosineBlox	
    f::Num
    a::Num
    phi::Vector
    offset::Num
    tstart::Num
    connector::Num
    odesystem::ODESystem
    function NoisyCosineBlox(;name, f=18, a=10, phi=0, offset=0, tstart=0)
        @named source = Cosine(frequency=f, amplitude=a, phase=phi, 		 
                        offset=offset, start_time=tstart, smooth=false)	
        new(f, a, phi, offset, tstart, source.output.u, source)
    end
end
const noisy_cosine_source = NoisyCosineBlox
