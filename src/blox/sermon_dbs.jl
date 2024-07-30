using ModelingToolkitStandardLibrary.Blocks: smooth_square, smooth_step
using IfElse

# TODO: List of missing components needed for this tutorial
# 1. SuperBlox of Kuramoto oscillators to collect dynamics
# 2. DBS stimulator block as implemented in the paper
# 4. BloxConnector functionality for neurotransmitter pools
# 5. BloxConnector functionality for DBS stimulator

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
                        f_stim=130,
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

