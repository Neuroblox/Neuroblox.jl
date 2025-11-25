recursive_getdefault(x) = x
function recursive_getdefault(x::Union{MTK.Num, MTK.BasicSymbolic})
    def_x = MTK.getdefault(x)
    vars = get_variables(def_x)
    defs = Dict(var => MTK.getdefault(var) for var in vars)
    substitute(def_x, defs)
end

function output end

function define_neuron(sys; mod=@__MODULE__(), generate_discrete_event_functions=true)
    T = typeof(sys)
    name = namespaced_nameof(sys)
    system = structural_simplify(sys.system; fully_determined=false)
    params = parameters(system)
    t = Symbol(get_iv(system))

    get_diff_state = @rule D(~s) => (~s).f
    r = (Postwalk ∘ Chain ∘ map)(unknowns(system)) do s
        (@rule s => s.f)
    end
    differentials = map(equations(system)) do eq
        (;lhs, rhs) = eq
        (;diff_state=toexpr(get_diff_state(lhs);), ex=toexpr(r(rhs)))
    end
    differentials = filter(x -> !isnothing(x.diff_state), differentials)
    
    inputs = [s for s ∈ unknowns(system) if  MTK.isinput(s)]
  
    p_syms = map(Symbol, params)
    s_syms = map(x -> x.diff_state, differentials)
    input_syms = map(x -> tosymbol(x; escape=false), inputs)

    p_and_s_syms = [s_syms; p_syms]

    rhss = map(x -> x.ex, differentials)
    input_init = NamedTuple{(input_syms...,)}(ntuple(i -> 0.0, length(inputs)))

    outs = NeurobloxBase.outputs(sys; namespaced=false)
    output_syms = map(x -> tosymbol(x; escape=false), outs)
    
    
    @eval mod begin
        $GraphDynamics.initialize_input(s::$Subsystem{$T}) = $input_init
        function $GraphDynamics.subsystem_differential((; $(p_and_s_syms...),)::$Subsystem{$T}, ($(input_syms...),), t)
            Dneuron = $SubsystemStates{$T}(
                $NamedTuple{$(Expr(:tuple, QuoteNode.(s_syms)...))}(
                    ($(rhss...),)
                )
            )
        end
        function $GraphDynamics.to_subsystem($name::$T)
            states = $SubsystemStates{$T}($NamedTuple{$(Expr(:tuple, QuoteNode.(s_syms)...))}(
                $(Expr(:tuple, (:(float($recursive_getdefault($getproperty(NeurobloxBase.get_system($name), $(QuoteNode(s)))))) for s ∈ s_syms)...))
            ))
            params = $SubsystemParams{$T}($NamedTuple{$(Expr(:tuple, QuoteNode.(p_syms)...))}(
                $(Expr(:tuple, (:($recursive_getdefault($getproperty($NeurobloxBase.get_system($name), $(QuoteNode(s))))) for s ∈ p_syms)...))
            ))
            $Subsystem(states, params)
        end

        function $NeurobloxBase.param_symbols(::Type{$T})
            $(Expr(:tuple, QuoteNode.(p_syms)...))
        end
        function $NeurobloxBase.state_symbols(::Type{$T})
            $(Expr(:tuple, QuoteNode.(s_syms)...))
        end
        function $NeurobloxBase.input_symbols(::Type{$T})
            $(Expr(:tuple, QuoteNode.(input_syms)...))
        end
        function $NeurobloxBase.output_symbols(::Type{$T})
            $(Expr(:tuple, QuoteNode.(output_syms)...))
        end
        function $NeurobloxBase.computed_state_symbols(::Type{$T})
            # This method doesn't support computed states
            ()
        end
    end
    if system isa SDESystem
        neqs = map(get_noiseeqs(system)) do eq
            (r(eq))
        end
        if any(row -> count(!iszero, row) > 1, eachrow(neqs))
            error("Attempted to construct subsystem with non-diagonal noise (i.e. the same noise parameter appears in multiple equations). This is not yet supported by GraphDynamics.jl")
        end
        neqs_diag = map(eachindex(s_syms)) do i
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
        @eval mod begin
            $GraphDynamics.isstochastic(::Type{<:$T}) = true
            Base.@propagate_inbounds function $GraphDynamics.apply_subsystem_noise!(v, (; $(p_and_s_syms...),)::$Subsystem{$T}, $t)
                $(Expr(:block, neq_gen...))
            end
        end
    end
    
    if length(outs) == 1
        out = only(outs)
        output_sym = hasproperty(out.val, :f) ? Symbol(out.val.f) : Symbol(out.val)
        @eval mod $GraphDynamicsInterop.output(s::$Subsystem{$T}) = s.$output_sym
    end
    
    if !isempty(get_continuous_events(system))
        cb = only(collect(get_continuous_events(system))) # currently only support single events
        cb_eqs = r(only(cb.eqs))
        ev_condition = Expr(:call, :-, toexpr(r(cb_eqs.lhs)), toexpr(r(cb_eqs.rhs)))
        cb_affects = map(r, cb.affect)
        
        ev_affect = :(NamedTuple{$(Expr(:tuple, map(x -> QuoteNode(Symbol(r(x.lhs))), cb_affects)...))}(
            $(Expr(:tuple, map(x -> toexpr(r(x.rhs)), cb_affects)...))
        ))
        @eval mod begin
            $GraphDynamics.has_continuous_events(::$Type{$T}) = true
            $GraphDynamics.continuous_event_condition((; $(p_and_s_syms...))::$Subsystem{$T}, t, _) = $ev_condition
            function $GraphDynamics.apply_continuous_event!(integrator, sview, pview, neuron::$Subsystem{$T}, _)
                (; $(p_and_s_syms...)) = neuron
                sview[] = $SubsystemStates{$T}($merge($NamedTuple($get_states(neuron)), $ev_affect))
            end
        end
    end
    if generate_discrete_event_functions && !isempty(get_discrete_events(system))
        cb = only(collect(get_discrete_events(system))) # currently only support single events
        cb_eq = r(cb.condition)
        if cb_eq.f ∉ (<, >, <=, >=)
            error("unsupported callback condition $cb_eq")
        end
        ev_condition = Expr(:call, cb_eq.f, toexpr.(r.(cb_eq.arguments))...)
        cb_affects = map(r, cb.affects)
        
        
        ev_affect = :($NamedTuple{$(Expr(:tuple, map(x -> QuoteNode(Symbol(r(x.lhs))), cb_affects)...))}(
            $(Expr(:tuple, map(x -> toexpr(r(x.rhs)), cb_affects)...))
        ))

        @eval mod begin
            $GraphDynamics.has_discrete_events(::Type{$T}) = true
            $GraphDynamics.discrete_event_condition((; $(p_and_s_syms...))::Subsystem{$T}, t, _) = $ev_condition
            function $GraphDynamics.apply_discrete_event!(integrator, sview, pview, neuron::$Subsystem{$T}, _)
                (; $(p_and_s_syms...)) = neuron
                sview[] = $SubsystemStates{$T}(merge($NamedTuple($get_states(neuron)), $ev_affect))
            end
        end
    end
end

