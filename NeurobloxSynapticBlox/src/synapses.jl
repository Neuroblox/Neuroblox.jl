mutable struct GABA_A_Synapse <: AbstractReceptor
    const name::Symbol
    const namespace::Union{Nothing, Symbol}
    const param_vals::NamedTuple
    system::Union{Nothing, ODESystem} # Value of nothing indicates that the System hasn't been generated yet
    function GABA_A_Synapse(; name, namespace=nothing, E_syn=-70, τ₁=0.1, τ₂=70.0, g=1.0)
        pvs = (;E_syn, τ₁, τ₂, g)
        new(name,  namespace, pvs, nothing)
    end
end

mutable struct Glu_AMPA_Synapse <: AbstractReceptor
    const name::Symbol
    const namespace::Union{Nothing, Symbol}
    const param_vals::NamedTuple
    system::Union{Nothing, ODESystem} # Value of nothing indicates that the System hasn't been generated yet
    function Glu_AMPA_Synapse(; name, namespace=nothing, E_syn=0.0, τ₁=0.1, τ₂=5.0, g=1.0)
        pvs = (;E_syn, τ₁, τ₂, g)
        new(name,  namespace, pvs, nothing)
    end
end

function make_symbolic_states(n::Union{GABA_A_Synapse, Glu_AMPA_Synapse})
    @variables begin
		G(t)=0.0 
        [output=true]
		z(t)=0.0
        G_asymp(t)
        [input=true]
	end
    (; G, z, G_asymp)
end

function make_system(s::Union{GABA_A_Synapse, Glu_AMPA_Synapse})
    (; E_syn, τ₁, τ₂, g) = params = make_symbolic_params(s)
    (; G, z, G_asymp)    = states = make_symbolic_states(s)
    eqs = [
		D(G) ~ -G/τ₂ + z,
        D(z) ~ -z/τ₁ + G_asymp,
	]
	System(eqs, t, collect(states), collect(params); name = get_name(s))
end

mutable struct Glu_AMPA_STA_Synapse <: AbstractReceptor
    const name::Symbol
    const namespace::Union{Nothing, Symbol}
    const param_vals::NamedTuple
    system::Union{Nothing, ODESystem} # Value of nothing indicates that the System hasn't been generated yet
    function Glu_AMPA_STA_Synapse(; name, namespace=nothing,
                                  E_syn=0.0, τ₁=0.1, τ₂=5.0, τ₃=2000.0, τ₄=0.1, kₛₜₚ=0.5, g=1.0)
        pvs = (; E_syn, τ₁, τ₂, τ₃, τ₄, kₛₜₚ, g)
        new(name,  namespace, pvs, nothing)
    end
end

function make_symbolic_states(n::Glu_AMPA_STA_Synapse)
    @variables begin
		G(t)=0.0 
        [output=true]
		z(t)=0.0
        Gₛₜₚ(t)=0.0
        zₛₜₚ(t)=0.0
        G_asymp_pre(t)
        [input = true]
        G_asymp_post(t)
        [input = true]
	end
    (; G, z, Gₛₜₚ, zₛₜₚ, G_asymp_pre, G_asymp_post)
end

function make_system(s::Glu_AMPA_STA_Synapse)
    (; E_syn, τ₁, τ₂, τ₃, τ₄, kₛₜₚ, g)              = params = make_symbolic_params(s)
    (; G, z, Gₛₜₚ, zₛₜₚ, G_asymp_pre, G_asymp_post) = states = make_symbolic_states(s)
    eqs = [
        D(G)    ~ -G/τ₂ + z,
        D(z)    ~ -z/τ₁ + G_asymp_pre,
        D(Gₛₜₚ) ~ -Gₛₜₚ/τ₃ + (zₛₜₚ/5)*(kₛₜₚ-Gₛₜₚ),
        D(zₛₜₚ) ~ -zₛₜₚ/τ₄ + G_asymp_post,
	]
	System(eqs, t, collect(states), collect(params); name = get_name(s))
end
