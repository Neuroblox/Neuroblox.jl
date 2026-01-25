"""
    NGNMM_theta(Cₑ=30, Cᵢ=30, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10, η_0ᵢ=0, v_synₑₑ=10,
                v_synₑᵢ=-10, v_synᵢₑ=10, v_synᵢᵢ=-10, alpha_invₑₑ=10,
                alpha_invₑᵢ=0.8, alpha_invᵢₑ=10, alpha_invᵢᵢ=0.8, kₑₑ=0, kₑᵢ=0.5,
                kᵢₑ=0.65, kᵢᵢ=0)

A next-generation neural mass model of coupled theta neuron populations. Parameter description and their values are reported in [1].
Each mass consists of a population of two neurons `a` and `b`, coupled using different synaptic terms ``g``. The entire expression of these is given by:

```math
    \\frac{a_e}{dt} = \\frac{1}{C_e}(b_e*(a_e-1) - (\\Delta_e/2)*((a_e+1)^2-b_e^2) - \\eta_{0e}*b_e*(a_e+1) - (v_{syn, ee}*g_{ee}+v_{syn, ei}*g_{ei})*(b_e*(a_e+1)) - (g_{ee}/2+g_{ei}/2)*(a_e^2-b_e^2-1)) \\\\
    \\frac{b_e}{dt} = \\frac{1}{C_e}*((b_e^2-(a_e-1)^2)/2 - \\Delta_e*b_e*(a_e+1) + (\\eta_{0e}/2)*((a_e+1)^2-b_e^2) + (v_{syn, ee}*(g_{ee}/2)+v_{syn, ei}*(g_{ei}/2))*((a_e+1)^2-b_e^2) - a_e*b_e*(g_{ee}+g_{ei})) \\\\
    \\frac{a_i}{dt} = \\frac{1}{C_i}(b_i*(a_i-1) - (\\Delta_i/2)*((a_i+1)^2-b_i^2) - \\eta_{0i}*b_i*(a_i+1) - (v_{syn, ie}*g_{ie}+v_{syn, ii}*g_{ii})*(b_i*(a_i+1)) - (g_{ie}/2+g_{ii}/2)*(a_i^2-b_i^2-1)) \\\\
    \\frac{b_i}{dt} = \\frac{1}{C_i}*((b_i^2-(a_i-1)^2)/2 - \\Delta_i*b_i*(a_i+1) + (\\eta_{0i}/2)*((a_i+1)^2-b_i^2) + (v_{syn, ie}*(g_{ie}/2)+v_{syn, ii}*(g_{ii}/2))*((a_i+1)^2-b_i^2) - a_i*b_i*(g_{ie}+g_{ii})) \\\\
    \\frac{g_ee}{dt} = \\alpha_{inv, ee} (\\frac{k_{ee}}{C_e \\pi} \\frac{1-a_e^2-b_e^2}{(1+2*a_e+a_e^2+b_e^2)} - g_{ee}) \\\\
    \\frac{g_ei}{dt} = \\alpha_{inv, ei} (\\frac{k_{ei}}{C_i \\pi} \\frac{1-a_i^2-b_i^2}{(1+2*a_i+a_i^2+b_i^2)} - g_{ei}) \\\\
    \\frac{g_ie}{dt} = \\alpha_{inv, ie} (\\frac{k_{ie}}{C_e \\pi} \\frac{1-a_e^2-b_e^2}{(1+2*a_e+a_e^2+b_e^2)} - g_{ie}) \\\\
    \\frac{g_ii}{dt} = \\alpha_{inv, ii} (\\frac{k_{ii}}{C_i \\pi} \\frac{1-a_i^2-b_i^2}{(1+2*a_i+a_i^2+b_i^2)} - g_{ii})
```
Alternatively this blox is called by `NextGenerationEI()`, but this is deprecated and will be removed in future updates.

Citations:
1. Byrne Á, O'Dea RD, Forrester M, Ross J, Coombes S. Next-generation neural mass and field modeling. J Neurophysiol. 2020 Feb 1;123(2):726-742. doi: 10.1152/jn.00406.2019.
"""
@blox struct NGNMM_theta(;name,namespace=nothing,
                         Cₑ=30.0,Cᵢ=30.0, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0,
                         v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0,
                         alpha_invₑᵢ=0.8, alpha_invᵢₑ=10.0, alpha_invᵢᵢ=0.8, kₑₑ=0.0, kₑᵢ=0.5,
                         kᵢₑ=0.65, kᵢᵢ=0.0) <: AbstractNeuralMass
    @params Cₑ=Cₑ Cᵢ=Cᵢ Δₑ=Δₑ Δᵢ=Δᵢ η_0ₑ=η_0ₑ η_0ᵢ=η_0ᵢ v_synₑₑ=v_synₑₑ v_synₑᵢ=v_synₑᵢ v_synᵢₑ=v_synᵢₑ v_synᵢᵢ=v_synᵢᵢ alpha_invₑₑ=alpha_invₑₑ alpha_invₑᵢ=alpha_invₑᵢ alpha_invᵢₑ=alpha_invᵢₑ alpha_invᵢᵢ=alpha_invᵢᵢ kₑₑ=kₑₑ kₑᵢ=kₑᵢ kᵢₑ=kᵢₑ kᵢᵢ=kᵢᵢ
    @states(
        aₑ=-0.6,
        bₑ=0.18,
        aᵢ=0.02,
        bᵢ=0.21,
        gₑₑ=0,
        gₑᵢ=0.23,
        gᵢₑ=0.26,
        gᵢᵢ=0.0,
    )
    @inputs
    @equations begin
        D(aₑ) = (1/Cₑ)*(bₑ*(aₑ-1) - (Δₑ/2)*((aₑ+1)^2-bₑ^2) - η_0ₑ*bₑ*(aₑ+1) - (v_synₑₑ*gₑₑ+v_synₑᵢ*gₑᵢ)*(bₑ*(aₑ+1)) - (gₑₑ/2+gₑᵢ/2)*(aₑ^2-bₑ^2-1))
        D(bₑ) = (1/Cₑ)*((bₑ^2-(aₑ-1)^2)/2 - Δₑ*bₑ*(aₑ+1) + (η_0ₑ/2)*((aₑ+1)^2-bₑ^2) + (v_synₑₑ*(gₑₑ/2)+v_synₑᵢ*(gₑᵢ/2))*((aₑ+1)^2-bₑ^2) - aₑ*bₑ*(gₑₑ+gₑᵢ))
        D(aᵢ) = (1/Cᵢ)*(bᵢ*(aᵢ-1) - (Δᵢ/2)*((aᵢ+1)^2-bᵢ^2) - η_0ᵢ*bᵢ*(aᵢ+1) - (v_synᵢₑ*gᵢₑ+v_synᵢᵢ*gᵢᵢ)*(bᵢ*(aᵢ+1)) - (gᵢₑ/2+gᵢᵢ/2)*(aᵢ^2-bᵢ^2-1))
        D(bᵢ) = (1/Cᵢ)*((bᵢ^2-(aᵢ-1)^2)/2 - Δᵢ*bᵢ*(aᵢ+1) + (η_0ᵢ/2)*((aᵢ+1)^2-bᵢ^2) + (v_synᵢₑ*(gᵢₑ/2)+v_synᵢᵢ*(gᵢᵢ/2))*((aᵢ+1)^2-bᵢ^2) - aᵢ*bᵢ*(gᵢₑ+gᵢᵢ))
        D(gₑₑ) = alpha_invₑₑ*((kₑₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gₑₑ)
        D(gₑᵢ) = alpha_invₑᵢ*((kₑᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gₑᵢ)
        D(gᵢₑ) = alpha_invᵢₑ*((kᵢₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gᵢₑ)
        D(gᵢᵢ) = alpha_invᵢᵢ*((kᵢᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gᵢᵢ)
    end
end
const NextGenerationEI = NGNMM_theta
