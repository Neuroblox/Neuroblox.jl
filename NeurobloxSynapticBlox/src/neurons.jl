mutable struct HHExci <: AbstractExciNeuron
    name::Symbol
    system::Union{Nothing, ODESystem} # Value of nothing indicates that the System hasn't been generated yet
    namespace::Union{Nothing, Symbol}
    param_vals::NamedTuple
    default_synapses::Dict{@NamedTuple{name::Symbol, T::Type, param_vals::NamedTuple}, Any}
	function HHExci(;
                    name, 
                    namespace=nothing,
                    E_syn=0.0, 
                    G_syn=3, 
                    I_bg=0.0,
                    τ=5
                    )
        pvs = (
            E_syn=E_syn, 
			G_Na = 52, 
			G_K  = 20,
			G_L = 0.1, 
			E_Na = 55,
			E_K = -90,
			E_L = -60,
			G_syn = G_syn, 
			V_shift = 10,
			V_range = 35,
			τ = τ,
			I_bg=I_bg,
			kₛₜₚ = 0.5,
            spikes = 0,
			spk_const = 1.127
        )
		new(name, nothing, namespace, pvs, Dict{@NamedTuple{name::Symbol, T::Type, param_vals::NamedTuple}, Any}())
	end
end

function make_symbolic_states(n::HHExci)
    @variables begin 
		V(t)=-65.00 
		n(t)=0.32 
		m(t)=0.05 
		h(t)=0.59 
		I_syn(t)
		[input=true] 
        I_in(t)
        [input=true]
		I_asc(t)
		[input=true]
        jcn(t) 
		[input=true]
        spikes_cumulative(t)=0.0
        spikes_window(t)=0.0
	end
    (; V, n, m, h, spikes_cumulative, spikes_window, I_syn, I_asc, I_in, jcn) 
end

function make_system(blox::HHExci)
    params = make_symbolic_params(blox)
    states = make_symbolic_states(blox)
    (;E_syn, G_Na, G_K, G_L, E_Na, E_K, E_L, G_syn,
     V_shift, V_range, τ, I_bg, kₛₜₚ, spikes, spk_const) = params
    (; V, n, m, h, spikes_cumulative, spikes_window, I_syn, I_asc, I_in, jcn) = states
	αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
	βₙ(v) = 0.125*exp(-(v+44)/80)
	αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
	βₘ(v) = 4*exp(-(v+55)/18)
	αₕ(v) = 0.07*exp(-(v+44)/20)
	βₕ(v) = 1/(1+exp(-(v+14)/10))
	ϕ = 5
	G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
	eqs = [
		D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn, 
		D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
		D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m),
		D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
        D(spikes_cumulative) ~ spk_const*G_asymp(V,G_syn),
        D(spikes_window) ~ spk_const*G_asymp(V,G_syn)
	]
	System(eqs, t, collect(states), collect(params); name = get_name(blox))
end

default_synapse(n::HHExci; E_syn=recursive_getdefault(n.E_syn), τ₁=0.1, τ₂=recursive_getdefault(n.τ), kwargs...) =
    Glu_AMPA_Synapse(;name=Symbol("$(n.name)_synapse"), E_syn, τ₁, τ₂, kwargs...)

mutable struct HHInhi <: AbstractInhNeuron
    name::Symbol
    system::Union{Nothing, ODESystem} # Value of nothing indicates that the System hasn't been generated yet
    namespace::Union{Nothing, Symbol}
    param_vals::NamedTuple
    default_synapses::Dict{@NamedTuple{name::Symbol, T::Type, param_vals::NamedTuple}, Any}
	function HHInhi(;
                    name, 
                    namespace = nothing, 
                    E_syn=-70.0,
                    G_syn=11.5,
                    I_bg=0.0,
                    τ=70
                    )
		ps = (
			E_syn=E_syn,
			G_Na = 52, 
			G_K  = 20, 
			G_L = 0.1, 
			E_Na = 55, 
			E_K = -90, 
			E_L = -60, 
			G_syn = G_syn, 
			V_shift = 0, 
			V_range = 35, 
			τ = τ,
			I_bg=I_bg, 
		)
		new(name, nothing, namespace, ps, Dict{@NamedTuple{name::Symbol, T::Type, param_vals::NamedTuple}, Any}())
	end
end

function make_symbolic_states(n::HHInhi)
    @variables begin 
		V(t)=-65.00 
		n(t)=0.32 
		m(t)=0.05 
		h(t)=0.59 
		I_syn(t)
		[input=true] 
		I_asc(t)
		[input=true]
		I_in(t)
		[input=true]
        jcn(t)
        [input=true]
	end
    (; V, n, m, h, I_syn, I_asc, I_in, jcn)
end

function make_system(blox::HHInhi)
    (; E_syn, G_Na, G_K, G_L, E_Na, E_K,
     E_L, G_syn, V_shift, V_range, τ, I_bg) = params = make_symbolic_params(blox)
    (; V, n, m, h, I_syn, I_asc, I_in, jcn) = states = make_symbolic_states(blox)
	αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
	βₙ(v) = 0.125*exp(-(v+48)/80)
    αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
	βₘ(v) = 4*exp(-(v+58)/18)
    αₕ(v) = 0.07*exp(-(v+51)/20)
	βₕ(v) = 1/(1+exp(-(v+21)/10))   	
	ϕ = 5
	eqs = [
		D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in+jcn, 
		D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n), 
		D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m), 
		D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h)
	]
    System(eqs, t, collect(states), collect(params); name = get_name(blox))
end

default_synapse(n::HHInhi; E_syn=recursive_getdefault(n.E_syn), τ₁=0.1, τ₂=recursive_getdefault(n.τ), kwargs...) =
    GABA_A_Synapse(;name=Symbol("$(n.name)_synapse"), E_syn, τ₁, τ₂, kwargs...)

synapse_set(n::Union{HHExci, HHInhi}) = getfield(n, :default_synapses)

function get_synapse!(blox_src; synapse_kwargs=(;), kwargs...)
    syn = get(kwargs, :synapse) do
        syn = default_synapse(blox_src; synapse_kwargs...)
        (; name, param_vals) = syn
        syn_used = get!(blox_src.default_synapses, (; name, T=typeof(syn), param_vals), syn)
        syn_used
    end
    syn
end
