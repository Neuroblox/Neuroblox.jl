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
                      V_E=0.0,
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
        p = paramscoping(V_E=V_E, C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        V_E, C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
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

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end

struct AdamINP <: AbstractAdamNeuron
    params
    system
    namespace

    function AdamINP(;name,
                      namespace=nothing,
                      V_I=-80,
                      C=1.0,
                      Eₙₐ=50,
                      ḡₙₐ=100, 
                      Eₖ=-100,
                      ḡₖ=80,
                      Eₗ=-67,
                      ḡₗ=0.05,
                      τᵢ=6,
                      Iₐₚₚ=0.1,
                      Iₙₒᵢₛₑ=0.0
    )
        p = paramscoping(V_I=V_I, C=C, Eₙₐ=Eₙₐ, ḡₙₐ=ḡₙₐ, Eₖ=Eₖ, ḡₖ=ḡₖ, Eₗ=Eₗ, ḡₗ=ḡₗ, Iₐₚₚ=Iₐₚₚ, Iₙₒᵢₛₑ=Iₙₒᵢₛₑ)
        V_I, C, Eₙₐ, ḡₙₐ, Eₖ, ḡₖ, Eₗ, ḡₗ, Iₐₚₚ, Iₙₒᵢₛₑ = p
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

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)

    end
end

# Threshold θ is set to -59 mV so that the total impulse of an average spike is about 1.0
struct AdamGlu <: AbstractNeurotransmitter
    params
    system
    namespace

    function AdamGlu(;name,
                      namespace=nothing,
                      Glu_max = 1.0,
                      τ_Glu=1.2,
                      θ=-59.0)

        p = paramscoping(Glu_max=Glu_max, τ_Glu=τ_Glu, θ=θ)
        Glu_max, τ_Glu, θ = p
        sts = @variables Glu(t)=0.0 [output=true] jcn(t) [input=true]

        eqs = [D(Glu) ~ Glu_max*heaviside(jcn - θ) - Glu/τ_Glu]
       
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
                      k_on=5,
                      k_off=0.0055,
                      k_r=0.0018,
                      k_d=0.0084,
                      k_unblock=5.4,
                      k_block=0.61,
                      α=0.0916,
                      β=0.0465
                      )
        
        p = paramscoping(k_on=k_on, k_off=k_off, k_r=k_r, k_d=k_d, k_unblock=k_unblock, k_block=k_block, α=α, β=β)
        k_on, k_off, k_r, k_d, k_unblock, k_block, α, β = p

        sts = @variables begin 
            Glu(t)
            [input=true] 
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
                D(C_B) ~ k_off*C_AB - 2*k_on*Glu*C_B
              ]

        sys = System(eqs, t, sts, p; name=name)

        new(p, sys, namespace)
    end
end