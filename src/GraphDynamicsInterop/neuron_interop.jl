##----------------------------------------------
## Neurons / Neural Mass
##----------------------------------------------

# neurons and neural mass models have no sub-components
components(blox::Union{NeuralMassBlox, AbstractNeuronBlox}) = (blox,)

recursive_getdefault(x) = x
function recursive_getdefault(x::Union{MTK.Num, MTK.BasicSymbolic})
    def_x = MTK.getdefault(x)
    vars = get_variables(def_x)
    defs = Dict(var => MTK.getdefault(var) for var in vars)
    substitute(def_x, defs)
end


function define_neurons()
    for (name, T) ∈ [(:hhne, :HHNeuronExciBlox)
                     (:hhni, :HHNeuronInhibBlox)
                     (:hhni_msn_adam, :HHNeuronInhib_MSN_Adam_Blox)
                     (:hhni_fsi_adam, :HHNeuronInhib_FSI_Adam_Blox)
                     (:hhne_stn_adam, :HHNeuronExci_STN_Adam_Blox)
                     (:hhni_GPe_adam, :HHNeuronInhib_GPe_Adam_Blox)
                     (:ngei, :NextGenerationEIBlox)
                     (:wc, :WilsonCowan)
                     (:ho, :HarmonicOscillator)
                     (:jr, :JansenRit)  # Note! Regular JansenRit can support delays, and I have not yet implemented this!
                     (:if, :IFNeuron)
                     (:lif, :LIFNeuron) 
                     (:qif, :QIFNeuron)
                     (:izh, :IzhikevichNeuron)
                     (:lif_exci, :LIFExciNeuron)
                     (:lif_inh, :LIFInhNeuron)
                     (:pexci, :PINGNeuronExci)
                     (:pinhib, :PINGNeuronInhib)
                     ]
        sys = getproperty(Neuroblox, T)(;name)
        system = structural_simplify(sys.system; fully_determined=false)
        params = get_ps(system)
        t = Symbol(get_iv(system))

        states = [s for s ∈ unknowns(system) if !MTK.isinput(s)]
        inputs = [s for s ∈ unknowns(system) if  MTK.isinput(s)]

        # states_unwrapped = map(x -> x.f, states)
        # inputs_unwrapped = map(x -> x.f, inputs)

        p_syms = map(Symbol, params)
        s_syms = map(x -> tosymbol(x; escape=false), states)
        input_syms = map(x -> tosymbol(x; escape=false), inputs)

        p_and_s_syms = [s_syms; p_syms]

        r = (MTK.SymbolicUtils.Postwalk ∘ ModelingToolkit.Chain ∘ map)(unknowns(system)) do s
            (@rule s => s.f)
        end
        rhss = map(equations(system)) do eq
            toexpr(r(eq.rhs))
        end

        input_init = NamedTuple{(input_syms...,)}(ntuple(i -> 0.0, length(inputs)))
        
        @eval begin
            issupported(::$T) = true
            GraphDynamics.initialize_input(s::Subsystem{$T}) = $input_init
            function GraphDynamics.subsystem_differential((; $(p_and_s_syms...),)::Subsystem{$T}, ($(input_syms...),), t)
                Dneuron = SubsystemStates{$T}(
                    NamedTuple{$(Expr(:tuple, QuoteNode.(s_syms)...))}(
                        ($(rhss...),)
                    )
                )
            end
            function to_subsystem($name::$T)
                states = SubsystemStates{$T}(NamedTuple{$(Expr(:tuple, QuoteNode.(s_syms)...))}(
                    $(Expr(:tuple, (:(float($recursive_getdefault($getproperty(Neuroblox.get_system($name), $(QuoteNode(s)))))) for s ∈ s_syms)...))
                ))
                params = SubsystemParams{$T}(NamedTuple{$(Expr(:tuple, QuoteNode.(p_syms)...))}(
                    $(Expr(:tuple, (:($recursive_getdefault($getproperty(Neuroblox.get_system($name), $(QuoteNode(s))))) for s ∈ p_syms)...))
                ))
                Subsystem(states, params)
            end
        end
        if system isa SDESystem
            neqs = map(get_noiseeqs(system)) do eq
                (r(eq))
            end
            if any(row -> count(!iszero, row) > 1, eachrow(neqs))
                error("Attempted to construct subsystem with non-diagonal noise (i.e. the same noise parameter appears in multiple equations). This is not yet supported by GraphDynamics.jl")
            end
            neqs_diag = map(eachindex(states)) do i
                j = findfirst(!iszero, @view neqs[i, :])
                if isnothing(j)
                    0.0
                else
                    toexpr(neqs[i,j])
                end
            end
            neq_gen = [:(setindex!(v, $(neqs_diag[i]), $i)) for i in eachindex(neqs_diag) if !isequal(neqs_diag[i], 0.0)]
            #TODO: apply_subsystem_noise! currently doesn't support noise dependant on inputs
            # I'm not sure this is a practical problem, but might be something we want to support
            # in the future.
            #TODO: We currently only support diagonal noise (that is, the noise source in one
            # equation can't depend on the noise source from another equation). This needs to be
            # generalized, but how to handle it best will require a lot of thought.
            @eval begin
                GraphDynamics.isstochastic(::$T) = true
                GraphDynamics.isstochastic(::Subsystem{$T}) = true
                GraphDynamics.isstochastic(::SubsystemStates{$T}) = true
                Base.@propagate_inbounds function GraphDynamics.apply_subsystem_noise!(v, (; $(p_and_s_syms...),)::Subsystem{$T}, $t)
                    $(Expr(:block, neq_gen...))
                end
            end
        end
        
        if hasproperty(sys, :output)
            output_sym = hasproperty(sys.output.val, :f) ? Symbol(sys.output.val.f) : Symbol(sys.output.val)
            @eval output(s::Subsystem{$T}) = s.$output_sym
        end
        
        if !isempty(get_continuous_events(system))
            cb = only(collect(get_continuous_events(system))) # currently only support single events
            cb_eqs = r(only(cb.eqs))
            ev_condition = Expr(:call, :-, toexpr(r(cb_eqs.lhs)), toexpr(r(cb_eqs.rhs)))
            cb_affects = map(r, cb.affect)
            
            ev_affect = :(NamedTuple{$(Expr(:tuple, map(x -> QuoteNode(Symbol(r(x.lhs))), cb_affects)...))}(
                $(Expr(:tuple, map(x -> toexpr(r(x.rhs)), cb_affects)...))
            ))
            @eval begin
                GraphDynamics.has_continuous_events(::Type{$T}) = true
                GraphDynamics.continuous_event_condition((; $(p_and_s_syms...))::Subsystem{$T}, t, _) = $ev_condition
                function GraphDynamics.apply_continuous_event!(integrator, sview, pview, neuron::Subsystem{$T}, _)
                    (; $(p_and_s_syms...)) = neuron
                    sview[] = SubsystemStates{$T}(merge(NamedTuple(get_states(neuron)), $ev_affect))
                end
            end
        end
    end
end
define_neurons() # it's useful when developing this module to have these in a function

#Maybe should just encorporate this into define_neurons()
# for T ∈ [:LIFExciNeuron, :LIFInhNeuron]
#     @eval begin
#         GraphDynamics.has_discrete_events(::Type{$T}) = true
#         GraphDynamics.discrete_event_condition((; t_refract_end)::Subsystem{$T}, t) = t_refract_end == t
#         function GraphDynamics.apply_discrete_event!(integrator, _, pview, neuron::Subsystem{$T}, _)
#             params = get_params(neuron)
#             pview[] = @set params.is_refractory = 0
#         end
#     end
# end

issupported(::PoissonSpikeTrain) = true
components(p::PoissonSpikeTrain) = (p,)
function to_subsystem(s::PoissonSpikeTrain)
    states = SubsystemStates{PoissonSpikeTrain, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{PoissonSpikeTrain}((;))
    Subsystem(states, params)
end
GraphDynamics.initialize_input(s::Subsystem{PoissonSpikeTrain}) = (;)
GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{PoissonSpikeTrain}, _, _) = nothing
GraphDynamics.subsystem_differential_requires_inputs(::Type{PoissonSpikeTrain}) = false


#-------------------------
# Kuramoto
issupported(::KuramotoOscillator) = true
function to_subsystem(o::KuramotoOscillator)
    states = SubsystemStates{KuramotoOscillator}((;θ=0.0,))
    params = SubsystemParams{KuramotoOscillator}((;ω=getdefault(o.system.ω), ζ=getdefault(o.system.ζ)))
    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{KuramotoOscillator}) = 0.0
function GraphDynamics.subsystem_differential(s::Subsystem{KuramotoOscillator}, jcn, t)
    SubsystemStates{KuramotoOscillator}((; #=D=#θ= s.ω + jcn))
end
GraphDynamics.isstochastic(::Type{KuramotoOscillator}) = true
function GraphDynamics.apply_subsystem_noise!(vstates, (;ζ,)::Subsystem{KuramotoOscillator}, t)
    vstates[1] = ζ
    nothing
end


#-------------------------
# Matrisome
issupported(::Matrisome) = true
components(m::Matrisome) = (m,)
GraphDynamics.initialize_input(s::Subsystem{Matrisome}) = 0.0
function GraphDynamics.apply_subsystem_differential!(_, m::Subsystem{Matrisome}, jcn, t)
    m.jcn_ref[] = jcn
end
function to_subsystem(s::Matrisome)
    states = SubsystemStates{Matrisome, Float64, NamedTuple{(), Tuple{}}}((;))
    params = SubsystemParams{Matrisome}((; H=1, TAN_spikes=0.0, jcn_=0.0, jcn_ref=Ref(0.0), H_=1, t_event=s.t_event + sqrt(eps(s.t_event))))
    #TODO: support observed variables ρ = H*jcn, ρ_ = H_*jcn_
    Subsystem(states, params)
end
GraphDynamics.has_discrete_events(::Type{Matrisome}) = true
GraphDynamics.discrete_events_require_inputs(::Type{Matrisome}) = true
function GraphDynamics.discrete_event_condition((;t_event,)::Subsystem{Matrisome}, t, _)
    t == t_event
end
GraphDynamics.event_times((;t_event)::Subsystem{Matrisome}) = t_event
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{Matrisome}, _, jcn)
    # recording the values of jcn and H at the event time in the parameters jcn_ and H_
    params = get_params(s)
    vparams[] = @set params.jcn_ = jcn
    nothing
end

#-------------------------
# Striosome
issupported(::Striosome) = true
components(s::Striosome) = (s,)
GraphDynamics.initialize_input(s::Subsystem{Striosome}) = 0.0
GraphDynamics.subsystem_differential(s::Subsystem{Striosome}, _, _) = SubsystemStates{Striosome}((;))
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{Striosome}, jcn, t)
    s.jcn_ref[] = jcn # We need to store the input to the Striosome because TAN and SNc blox can view it
    nothing
end
function to_subsystem(s::Striosome)
    states = SubsystemStates{Striosome, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{Striosome}((; H=1, H_learning=1.0, jcn_ref=Ref(0.0)))
    #TODO: support observed variable ρ = H*jcn
    Subsystem(states, params)
end


#-------------------------
# TAN
issupported(::TAN) = true
components(t::TAN) = (t,)
GraphDynamics.initialize_input(s::Subsystem{TAN}) = 0.0
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{TAN}, jcn, t)
    nothing
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{TAN}) = false
function to_subsystem(s::TAN)
    κ = getdefault(s.system.κ)
    λ = getdefault(s.system.λ)
    states = SubsystemStates{TAN, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{TAN}((; κ, λ))
    #TODO: support observed variable R = min(κ, κ/(λ*jcn + sqrt(eps())))
    Subsystem(states, params)
end

#-------------------------
# SNc
issupported(::SNc) = true
components(s::SNc) = (s,)
GraphDynamics.initialize_input(s::Subsystem{SNc}) = (;jcn,)
GraphDynamics.subsystem_differential(s::Subsystem{SNc}, _, _) = SubsystemStates{SNc}((;))
function GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{SNc}, args...)
    nothing
end
GraphDynamics.subsystem_differential_requires_inputs(::Type{SNc}) = false
function to_subsystem(s::SNc)
    (;N_time_blocks, κ_DA, DA_reward) = s
    
    κ = getdefault(s.system.κ)
    λ = getdefault(s.system.λ)
    states = SubsystemStates{SNc, Float64, @NamedTuple{}}((;))
    params = SubsystemParams{TAN}((;κ_DA, N_time_blocks, DA_reward, λ_DA, t_event=t_event+sqrt(eps(t_event)), jcn_=0.0))
    #TODO: support observed variables R  ~ min(κ_DA, κ_DA/(λ_DA*jcn  + sqrt(eps())))
    #                                 R_ ~ min(κ_DA, κ_DA/(λ_DA*jcn_ + sqrt(eps())))
    Subsystem(states, params)
end

GraphDynamics.has_discrete_events(::Type{SNc}) = true
GraphDynamics.discrete_events_require_inputs(::Type{SNc}) = true
function GraphDynamics.discrete_event_condition((;t_event,)::Subsystem{SNc}, t, _)
    t == t_event
end
GraphDynamics.event_times((;t_event)::Subsystem{SNc}) = t_event
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{SNc}, _, jcn)
    # recording the values of jcn and H at the event time in the parameters jcn_ and H_
    params = get_params(s)
    vparams[] = @set params.jcn_ = jcn
    nothing
end
