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
        
        sts = @variables n_pool(t)=0.0 [output = true] jcn(t)=0.0 [input=true]
        eqs = [D(n_pool) ~ (1-n_pool)/τ_pool - p_pool*n_pool*jcn]
        sys = System(eqs, t, sts, p; name=name)
        new(p, sts[1], sts[2], sys, namespace)
    end
end
