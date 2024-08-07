using IfElse

# TODO: List of missing components needed for this tutorial
# 1. SuperBlox of Kuramoto oscillators to collect dynamics

# Create neurotransmitter pools 
struct SermonNPool <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function SermonNPool(;
                        name,
                        namespace=nothing,
                        τ_pool=0.1,
                        p_pool=0.3
            )
        p = paramscoping(τ_pool=τ_pool, p_pool=p_pool)
        τ_pool, p_pool = p
        
        sts = @variables n_pool(t)=1.0 [output = true] jcn(t)=0.0 [input=true]
        eqs = [D(n_pool) ~ (1-n_pool)/τ_pool - p_pool*n_pool*jcn]
        sys = System(eqs, t, sts, p; name=name)
        new(p, sts[1], sts[2], sys, namespace)
    end
end

# Create a new DBS stimulator block
# Largely based on MTK Standard Library smooth_square function in src/Blocks/sources.jl
struct SermonDBS <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function SermonDBS(;
                        name,
                        namespace=nothing,
                        f_stim=130.0,
                        amplitude=1.0,
                        pulse_width=0.0001,
                        start_time=0.0      
            )
        p = paramscoping(f_stim=f_stim, amplitude=amplitude, pulse_width=pulse_width, start_time=start_time)
        f_stim, amplitude, pulse_width, start_time = p
        
        sts = @variables u(t)=0.0 [output = true]
        eqs = [u ~ IfElse.ifelse(t > start_time, IfElse.ifelse(t % (1.0/f_stim) < pulse_width, amplitude, 0), 0)]
        sys = System(eqs, t, sts, p; name=name)
        new(p, sts[1], nothing, sys, namespace)
    end
end

function (bc::BloxConnector)(
    bloxout::SermonDBS, 
    bloxin::SermonNPool; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    x = namespace_expr(bloxout.output, sys_out)

    eq = sys_in.jcn ~ w*x
    accumulate_equation!(bc, eq)
end

# Redfine Kuramoto oscillator connections with arbitrary connection rule
function (bc::BloxConnector)(
    bloxout::KuramotoOscillator, 
    bloxin::KuramotoOscillator; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    #technically these two lines aren't needed, but useful to have if weighting occurs here
    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    if haskey(kwargs, :sermon_rule)
        if haskey(kwargs, :extra_bloxs) && haskey(kwargs, :extra_params)
            RRP, RP, RtP = kwargs[:extra_bloxs]
            out_RRP = namespace_expr(RRP.output, get_namespaced_sys(RRP))
            out_RP = namespace_expr(RP.output, get_namespaced_sys(RP))
            out_RtP = namespace_expr(RtP.output, get_namespaced_sys(RtP))

            M_RRP, M_RP, M_RtP, k_μ = kwargs[:extra_params]
        else
            error("Extra states and parameters need to be specified to use the Sermon rule")
        end
        xₒ = namespace_expr(bloxout.output, sys_out)
        xᵢ = namespace_expr(bloxin.output, sys_in) #needed because this is also the θ term of the block receiving the connection
        
        # Custom values for the connection rule
        f₀ = -0.780
        f₁ = 0.198
        f₂ = 0.302
        f₃ = 0.851
        f₄ = 0.998
        x = xₒ - xᵢ
        eq = sys_in.jcn ~ w * max(M_RRP * out_RRP, M_RP * out_RP, M_RtP * out_RtP)*(f₀ + f₁*cos(x) + f₂*sin(x) + f₃*cos(2*x) + f₄*sin(2*x))
        accumulate_equation!(bc, eq)
    end
end

function (bc::BloxConnector)(
    bloxout::SermonDBS, 
    bloxin::KuramotoOscillator; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    xₒ = namespace_expr(bloxout.output, sys_out)
    xᵢ = namespace_expr(bloxin.output, sys_in) #needed because this is also the θ term of the block receiving the connection

    eq = sys_in.jcn ~ w*xₒ*sin(xᵢ)
    accumulate_equation!(bc, eq)
end
