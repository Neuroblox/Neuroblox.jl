using IfElse

abstract type AbstractAdamNeuron <: AbstractNeuronBlox end

abstract type AbstractReceptor <: AbstractBlox end
abstract type AbstractNeurotransmitter <: AbstractBlox end

# Custom IfElse function to ensure differentiability so the solvers don't complain
function heaviside(x)
    IfElse.ifelse(x > 0, 1.0, 0.0)
end

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
                      Iₐₚₚ=-0.25,
                      Iₙₒᵢₛₑ=0.0)
        p = paramscoping(C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
        sts = @variables V(t)=0.0 m(t)=0.0 h(t)=0.0 n(t)=0.0 jcn(t) [input=true]

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

        eqs = [D(V) ~ (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
               D(m) ~ (m∞(V) - m)/τₘ(V),
               D(h) ~ (h∞(V) - h)/τₕ(V),
               D(n) ~ (n∞(V) - n)/τₙ(V)
        ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end

struct AdamIN <: AbstractAdamNeuron
    params
    system
    namespace

    function AdamIN(;name,
                      namespace=nothing,
                      C=1.0,
                      Eₙₐ=50,
                      ḡₙₐ=100, 
                      Eₖ=-100,
                      ḡₖ=80,
                      Eₗ=-67,
                      ḡₗ=0.05,
                      Iₐₚₚ=0.1,
                      Iₙₒᵢₛₑ=0.0
    )
        p = paramscoping(C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
        sts = @variables V(t)=0.0 m(t)=0.0 h(t)=0.0 n(t)=0.0 [output=true] jcn(t) [input=true]

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

        eqs = [D(V) ~ (Iₐₚₚ + Iₙₒᵢₛₑ - ḡₙₐ*m^3*h*(V - Eₙₐ) - ḡₖ*n^4*(V - Eₖ) - ḡₗ*(V - Eₗ) - jcn)/C,
               D(m) ~ (m∞(V) - m)/τₘ(V),
               D(h) ~ (h∞(V) - h)/τₕ(V),
               D(n) ~ (n∞(V) - n)/τₙ(V)
        ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end

struct AdamAMPA <: AbstractReceptor
    params
    system
    namespace

    function AdamAMPA(;
        name,
        namespace=nothing,
        g=0.2, # mS / cm⁻²
        E=0,
        τₑ=1.5
    )

        p = paramscoping(g=g, E=E, τₑ=τₑ)
        g, E, τₑ = p
        sts = @variables V(t) [input=true] sₐₘₚₐ(t)=0.0 [output=true]

        gₐₘₚₐ(v) = 5*(1+tanh(v/4))

        eqs = [
            D(sₐₘₚₐ) ~ gₐₘₚₐ(V)*(1-sₐₘₚₐ) - sₐₘₚₐ/τₑ
        ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end

struct AdamGABA <: AbstractReceptor
    params
    system
    namespace

    function AdamGABA(;
        name,
        namespace=nothing,
        g=5, # mS / cm⁻²
        E=-80,
        τᵢ=6
    )

    p = paramscoping(g=g, E=E, τᵢ=τᵢ)
    g, E, τᵢ = p
    sts = @variables V(t) [input=true] sᵧ(t)=0.0 [output=true]

    gᵧ(v) = 2*(1+tanh(v/4))

    eqs = [
            D(sᵧ) ~ gᵧ(V)*(1-sᵧ) - sᵧ/τᵢ
        ]
    
    sys = System(eqs, t, sts, p; name=name)

    new(p, sys, namespace)
    end
end

struct AdamNMDAR <: AbstractReceptor
    params
    system
    namespace

    function AdamNMDAR(;name,
                      namespace=nothing,
                      g=8.5, # mS / cm⁻²
                      E=0,
                      k_on=5,
                      k_off=0.0055,
                      k_r=0.0018,
                      k_d=0.0084,
                      k_unblock=5.4,
                      k_block=0.61,
                      α=0.0916,
                      β=0.0465,
                      Glu_max = 1.0,
                      τ_Glu=1.2,
                      θ=-59.0 # θ is set to -59 mV so that the total impulse of an average spike is about 1.0
    )
        
        p = paramscoping(g=g, E=E, k_on=k_on, k_off=k_off, k_r=k_r, k_d=k_d, k_unblock=k_unblock, k_block=k_block, α=α, β=β, Glu_max=Glu_max, τ_Glu=τ_Glu, θ=θ)
        g, E, k_on, k_off, k_r, k_d, k_unblock, k_block, α, β, Glu_max, τ_Glu, θ = p

        sts = @variables begin 
            V(t)
            [input=true] 
            C(t)=0.5
            C_A(t)=0.0
            C_AA(t)=0.0
            D_AA(t)=0.0
            O_AA(t)=0.0
            [output = true] 
            O_AAB(t)=0.0
            C_AAB(t)=0.0
            D_AAB(t)=0.0
            C_AB(t)=0.0
            C_B(t)=0.5    
            Glu(t)=0.0 
            jcn(t)
            [input=true]
        end

        eqs = [
                D(C) ~  k_off*C_A - 2*k_on*Glu*C,
                D(C_A) ~ 2*k_off*C_AA +  2*k_on*Glu*C - (k_on*Glu + k_off)*C_A,
                D(C_AA) ~ k_on*Glu*C_A + α*O_AA + k_r*D_AA - (2*k_off + β + k_d)*C_AA,
                D(D_AA) ~ k_d*C_AA - k_r*D_AA,
                D(O_AA) ~ β*C_AA + k_unblock*exp(V/47)*O_AAB - (α + k_block*exp(-V/17))*O_AA,
                D(O_AAB) ~ k_block*exp(-V/17)*O_AA + β*C_AAB - (k_unblock*exp(V/47) + α)*O_AAB,
                D(C_AAB) ~ α*O_AAB + k_on*Glu*C_AB + k_r*D_AAB - (β + 2*k_off + k_d)*C_AAB,
                D(D_AAB) ~ k_d*C_AAB - k_r*D_AAB,
                D(C_AB) ~ 2*k_off*C_AAB + 2*k_on*Glu*C_B - (k_on*Glu + k_off)*C_AB,
                D(C_B) ~ k_off*C_AB - 2*k_on*Glu*C_B,
                D(Glu) ~ Glu_max*heaviside(jcn - θ) - Glu/τ_Glu
              ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end