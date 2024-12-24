# Lead field function for LFPs
struct LeadField <: ObserverBlox
    params
    system
    namespace

    function LeadField(;name, namespace=nothing, L=1.0)
        p = paramscoping(L=L)
        L, = p

        sts = @variables lfp(t)=0.0 [irreducible=true, output=true, description="measurement"] jcn(t)=1.0 [input=true]

        eqs = [
            lfp ~ L * jcn
        ]

        sys = System(eqs, t; name=name)
        new(p, sys, namespace)
    end
end