abstract type AbstractAdamNeuron <: AbstractNeuronBlox end

struct AdamPYR <: AbstractAdamNeuron
    params
    system
    namespace

    function AdamPYR(;name,
                      namespace=nothing,
                      C=1.0,
                      Eₙₐ=50,
                      ḡₙₐ=100, 
                      Eₖ=-100,
                      ḡₖ=80,
                      Eₗ=-67,
                      ḡₗ=0.05,
                      τₑ=1.5,
                      Iₐₚₚ=-0.25,
                      Iₙₒᵢₛₑ=0.0)
        p = paramscoping(C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
        sts = @variables V(t)=0.0 m(t)=0.0 h(t)=0.0 n(t)=0.0 sₐₘₚₐ(t)=0.0 [output=true] jcn(t) [input=true]

        αₘ(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
        βₘ(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
        αₕ(v) = 0.128*exp((v+50.0)/18.0)
        βₕ(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
        αₙ(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
        βₙ(v) = 0.5*exp(-(v+57.0)/40.0)

        m∞(v) = αₘ(v)/(αₘ(v) + βₘ(v))
        h∞(v) = αₕ(v)/(αₕ(v) + βₕ(v))
        n∞(v) = αₙ(v)/(αₙ(v) + βₙ(v))

        τₘ(v) = 1.0/(αₘ(v) + βₘ(v))
        τₕ(v) = 1.0/(αₕ(v) + βₕ(v))
        τₙ(v) = 1.0/(αₙ(v) + βₙ(v))

        gₐₘₚₐ(v) = 5*(1+tanh(v/4))

        eqs = [D(V) ~ (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
               D(m) ~ (m∞(V) - m)/τₘ(V),
               D(h) ~ (h∞(V) - h)/τₕ(V),
               D(n) ~ (n∞(V) - n)/τₙ(V),
               D(sₐₘₚₐ) ~ gₐₘₚₐ(V)*(1-sₐₘₚₐ) - sₐₘₚₐ/τₑ
        ]

        sys = ODESystem(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end

struct AdamINP <: AbstractAdamNeuron
    params
    system
    namespace

    function AdamINP(;name,
                      namespace=nothing,
                      C=1.0,
                      Eₙₐ=50,
                      ḡₙₐ=100, 
                      Eₖ=-100,
                      ḡₖ=80,
                      Eₗ=-67,
                      ḡₗ=0.05,
                      τᵢ=6,
                      Iₐₚₚ=0.1,
                      Iₙₒᵢₛₑ=0.0)
        p = paramscoping(C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
        sts = @variables V(t)=0.0 m(t)=0.0 h(t)=0.0 n(t)=0.0 sᵧ(t)=0.0 [output=true] jcn(t) [input=true]

        αₘ(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
        βₘ(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
        αₕ(v) = 0.128*exp((v+50.0)/18.0)
        βₕ(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
        αₙ(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
        βₙ(v) = 0.5*exp(-(v+57.0)/40.0)

        m∞(v) = αₘ(v)/(αₘ(v) + βₘ(v))
        h∞(v) = αₕ(v)/(αₕ(v) + βₕ(v))
        n∞(v) = αₙ(v)/(αₙ(v) + βₙ(v))

        τₘ(v) = 1.0/(αₘ(v) + βₘ(v))
        τₕ(v) = 1.0/(αₕ(v) + βₕ(v))
        τₙ(v) = 1.0/(αₙ(v) + βₙ(v))

        gᵧ(v) = 2*(1+tanh(v/4))

        eqs = [D(V) ~ (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
               D(m) ~ (m∞(V) - m)/τₘ(V),
               D(h) ~ (h∞(V) - h)/τₕ(V),
               D(n) ~ (n∞(V) - n)/τₙ(V),
               D(sᵧ) ~ gᵧ(V)*(1-sᵧ) - sᵧ/τᵢ
        ]

        sys = ODESystem(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end