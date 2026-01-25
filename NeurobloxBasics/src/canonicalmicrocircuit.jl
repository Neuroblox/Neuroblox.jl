"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
@blox struct JansenRitSPM(;name, namespace=nothing, τ=1.0, r=2/3) <: AbstractNeuralMass
    @params τ r
    @states x=0.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = (-2*y - x/τ + jcn)/τ
    end
end

struct CanonicalMicroCircuit <: AbstractComposite
    name
    namespace
    parts
    graph

    function CanonicalMicroCircuit(;name, namespace=nothing, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        g = GraphSystem(name=namespaced_name(namespace, name))
        @graph! g begin
            @nodes begin
                ss = JansenRitSPM(;τ=τ_ss, r=r_ss)  # spiny stellate
                sp = JansenRitSPM(;τ=τ_sp, r=r_sp)  # superficial pyramidal
                ii = JansenRitSPM(;τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
                dp = JansenRitSPM(;τ=τ_dp, r=r_dp)  # deep pyramidal
            end
            sblox_parts = vcat(ss, sp, ii, dp)
            @connections begin 
                ss => ss, [weight = -800.0]
                sp => ss, [weight = -800.0]
                ii => ss, [weight = -1600.0]
                ss => sp, [weight =  800.0]
                sp => sp, [weight = -800.0]
                ss => ii, [weight =  800.0]
                ii => ii, [weight = -800.0]
                dp => ii, [weight =  400.0]
                ii => dp, [weight = -400.0]
                dp => dp, [weight = -200.0]
            end
        end
        new(name, namespace, sblox_parts, g)
    end
end
