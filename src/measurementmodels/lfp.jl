# Lead field function for LFPs
struct LeadField <: ObserverBlox
    params
    output
    jcn
    odesystem
    namespace

    function LeadField(;name, namespace=nothing, L=1.0)
        p = paramscoping(L=L)
        L, = p

        sts = @variables lfp(t)=0.0 [irreducible=true, output=true, description="measurement"] jcn(t)=1.0 [input=true]

        eqs = [
            lfp ~ L * jcn
        ]

        sys = System(eqs, t, sts, p; name=name)
        new(p, Num(0), sts[2], sys, namespace)
    end
end