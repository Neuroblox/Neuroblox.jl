function get_params_nt(x)
    getfield(x, :params_nt)
end

function fwd_getproperty(blox::T, prop) where {T}
    if prop == :name
        namespaced_nameof(blox)
    elseif (prop ∈ param_symbols(T)) || (prop ∈ state_symbols(T)) || (prop ∈ computed_property_symbols(T)) || (prop ∈ computed_property_with_inputs_symbols(T)) || (prop ∈ input_symbols(T))
        namespaced_name(namespaced_name(namespaceof(blox), getfield(blox, :name)), prop)
    else
        getfield(blox, prop)
    end
end

function set_fn_lnn!(fdef, lnn::LineNumberNode)
    if isexpr(fdef, :function) || isexpr(fdef, :(=)) || isexpr(fdef, Symbol("->"))
        fdef.args[2].args[1] = lnn
    else
        error(ArgumentError("Cannot set the linenumber node, expected a function definition, got $fdef"))
    end
end

"""
    @blox struct StructName(; name, namespace, kwargs...) <: SuperType
        @params        # Define parameters (e.g., `@params C Eₘ Rₘ`)
        @states        # Define states with initial values (e.g., `@states V=-70.0 G=0.0`)
        @inputs         # Define inputs with zero/initial values (e.g., `@inputs jcn=0.0`)
        @outputs        # Declare outputs (e.g., `@outputs V G`)
        @equations      # Define differential equations (e.g., `@equations begin D(V) = ... end`)
        [@noise_equations]   # Optional: Define stochastic noise terms
        [@discrete_events]   # Optional: Define discrete events (condition => effect)
        [@continuous_events] # Optional: Define continuous events (condition => event)
        [@event_times]       # Optional: Specify event times
        [@extra_fields]      # Optional: Add arbitrary fields
    end

Define a Neuroblox-compatible struct for dynamic graph simulations.
Automatically generates interfaces for GraphDynamics.jl and NeurobloxBase.
Requirements:

`@params`, `@states`, `@inputs`, `@outputs`, and `@equations` are mandatory.
Constructor must include `name` and `namespace` as keyword arguments.
Equations must cover all states and use `D(state) = expression` syntax.

Optional:

`@noise_equations`, `@discrete_events`, `@continuous_events`, `@event_times`, `@extra_fields`.

Generated:

Struct type, property access, and standard interfaces for simulation.
Default constructor initializes parameters, states, inputs, and extra fields.

For more information, see the [documentation](https://neuroblox.github.io/NeurobloxDocsHost/) section on Defining your own Blox with `@blox`
"""
macro blox(structdef)
    # Parse struct header
    if !isexpr(structdef, :struct)
        error(ArgumentError("Inputs to @blox must be struct declarations"))
    end
    
    (ismutable, struct_header, struct_body) = structdef.args
    
    if isexpr(struct_header, :(<:))
        sig, suptype = struct_header.args
    elseif isexpr(struct_header, :call)
        sig = struct_header
        suptype = Any
    else
        error(ArgumentError("Inputs to @blox must be struct declarations where the struct signature is a constructor signature"))
    end
    
    if !isexpr(sig, :call)
        error(ArgumentError("Inputs to @blox must be struct declarations where the struct signature is a constructor signature"))
    end
    
    (blox_name, blox_args...) = sig.args
    
    # Initialize tracking variables
    lnns = LineNumberNode[]
    local param_syms, param_vals_ex, params_lnn
    local state_data, state_syms, states_lnn
    local inputs_data, input_syms, inputs_lnn
    local outputs_syms, outputs_lnn
    local equations_lnn, equations_inner_lnns, equations_parsed
    local disc_cond, disc_affects
    local cont_cond, cont_affects
    local ev_time_arg
    local noise_eqs_parsed, noise_eqs_inner_lnns
    local comp_prop_data
    local comp_prop_inputs_data

    should_emit_equations = true
    extra_field_data = @NamedTuple{field_name, field_val}[]
    constructor_junk = Expr(:block)
    equations_junk = Expr(:block)
    
    # Parse struct body
    for item ∈ struct_body.args
        if item isa LineNumberNode
            push!(lnns, item)
        elseif isexpr(item, :macrocall)
            item_name, item_lnn, item_args... = item.args
            
            if item_name == Symbol("@extra_fields")
                extra_field_data = parse_extra_fields(item_args)
                
            elseif item_name == Symbol("@params")
                param_syms, param_vals_ex = parse_params(item_args)
                params_lnn = lnns[end]
                
            elseif item_name == Symbol("@states")
                state_data, state_syms = parse_states(item_args)
                states_lnn = lnns[end]
                
            elseif item_name == Symbol("@inputs")
                inputs_data, input_syms = parse_inputs(item_args)
                inputs_lnn = lnns[end]
                
            elseif item_name == Symbol("@outputs")
                outputs_syms = parse_outputs(item_args)
                outputs_lnn = lnns[end]
                
            elseif item_name == Symbol("@computed_states")
                error("Computed states are not currently supported by @blox")
                
            elseif item_name == Symbol("@discrete_events")
                disc_cond, disc_affects = parse_discrete_events(item_args)

            elseif item_name == Symbol("@continuous_events")
                cont_cond, cont_affects = parse_continuous_events(item_args)
                
            elseif item_name == Symbol("@event_times")
                ev_time_arg = only(item_args)
                
            elseif item_name == Symbol("@equations")
                if item.args[1] == QuoteNode(:skip)
                    should_emit_equations = false
                else
                    equations_parsed, equations_inner_lnns, equations_junk = 
                        parse_equations(item_args, param_syms, state_syms, inputs_data)
                    equations_lnn = lnns[end]
                end
            elseif item_name == Symbol("@noise_equations")
                noise_eqs_parsed, noise_eqs_inner_lnns = 
                    parse_noise_equations(item_args, param_syms, state_syms, inputs_data)

            elseif item_name == Symbol("@computed_properties")
                comp_prop_data = parse_computed_properties(lnns[end], item_args, param_syms, state_syms)
                
            elseif item_name == Symbol("@computed_properties_with_inputs")
                comp_prop_inputs_data = parse_computed_properties(lnns[end], item_args, param_syms, state_syms)
            else
                push!(constructor_junk.args, lnns[end], item)
            end
        else
            push!(constructor_junk.args, lnns[end], item)
        end
    end
    
    # Generate output code
    out = Expr(:block)
    out_args = out.args
    
    # Struct definition
    push!(out_args, emit_struct_definition(blox_name, suptype, sig, param_syms, 
                                             extra_field_data, param_vals_ex, 
                                             constructor_junk, __source__))
    
    # Getproperty
    push!(out_args, emit_getproperty(blox_name, __source__))
    
    # Params
    push!(out_args, emit_param_symbols(blox_name, param_syms, params_lnn))
    
    # States
    push!(out_args, emit_state_symbols(blox_name, state_syms, states_lnn))
    
    # Inputs
    push!(out_args, emit_input_symbols(blox_name, inputs_data, inputs_lnn))
    
    # Outputs
    if @isdefined(outputs_syms)
        push!(out_args, emit_output_symbols(blox_name, outputs_syms))
    end
    
    
    # Event times
    if @isdefined(ev_time_arg)
        push!(out_args, emit_event_times(blox_name, param_syms, state_syms, ev_time_arg))
    end
    
    # Discrete events
    if @isdefined(disc_cond)
        push!(out_args, emit_discrete_events(blox_name, param_syms, state_syms, 
                                             disc_cond, disc_affects))
    end

    # Discrete events
    if @isdefined(cont_cond)
        push!(out_args, emit_continuous_events(blox_name, param_syms, state_syms, 
                                               cont_cond, cont_affects))
    end
    
    # Subsystem methods
    push!(out_args, emit_subsystem_methods(blox_name, param_syms, state_data, state_syms,
                                           input_syms, equations_parsed, equations_junk,
                                           should_emit_equations,
                                           params_lnn, states_lnn, __source__))
    
    # Stochastic noise
    if @isdefined(noise_eqs_parsed)
        push!(out_args, emit_stochastic_methods(blox_name, param_syms, state_syms,
                                                noise_eqs_parsed, noise_eqs_inner_lnns))
    end

    if @isdefined(comp_prop_data)
        push!(out_args, emit_computed_properties(blox_name, param_syms, state_syms, comp_prop_data))
    end

    if @isdefined(comp_prop_inputs_data)
        push!(out_args,
              emit_computed_properties_with_inputs(blox_name, param_syms, state_syms, input_syms, comp_prop_inputs_data))
    end
    
    
    out
end

function parse_extra_fields(args)
    extra_field_data = @NamedTuple{field_name, field_val}[]
    for arg ∈ args
        if isexpr(arg, :(=), 2)
            field_name, field_val = arg.args
            push!(extra_field_data, (; field_name, field_val))
        else
            error("Invalid extra field provided to @extra_fields, expected an assignment expression got $arg")
        end
    end
    extra_field_data
end

function parse_params(args)
    param_syms = map(args) do ex
        if ex isa Symbol
            ex
        elseif isexpr(ex, :(=))
            ex.args[1]
        else
            error(ArgumentError("Invalid argument $ex to @params. Arguments must be either symbols, or expressions of the form `sym = val`"))
        end
    end
    param_vals_ex = :((; $(args...),))
    (param_syms, param_vals_ex)
end

function parse_states(args)
    state_data = map(args) do ex
        if isexpr(ex, :(=))
            sym, val = ex.args
            (sym, val)
        else
            error(ArgumentError("Invalid argument $ex to @states. Arguments must be expressions of the form `sym = val`."))
        end
    end
    state_syms = [sym for (sym, _) ∈ state_data]
    (state_data, state_syms)
end

function parse_inputs(args)
    inputs_data = map(args) do ex
        if isexpr(ex, :(=))
            sym, val = ex.args
            (sym, val)
        else
            error(ArgumentError("Invalid argument $ex to @inputs. Arguments must be expressions of the form `sym = initial_val`."))
        end
    end
    input_syms = first.(inputs_data)
    (inputs_data, input_syms)
end

function parse_outputs(args)
    outputs_syms = map(args) do ex
        if ex isa Symbol
            ex
        else
            error(ArgumentError("@outputs declaration must be a simple list of output symbols, got $ex."))
        end
    end
    outputs_syms
end

function parse_equations(args, param_syms, state_syms, input_data)
    if !(@isdefined(param_syms)) && !(@isdefined(state_syms)) && !(@isdefined(input_data))
        error(ArgumentError("@equations must be declared after @parameters, @states, and @inputs"))
    end
    
    if length(args) > 1
        error(ArgumentError("@equations expects a single begin...end block"))
    end
    if !isexpr(args[1], :block)
        error(ArgumentError("@equations expects a begin...end block"))
    end
    
    equations_inner_lnns = LineNumberNode[]
    equations_parsed = @NamedTuple{var::Symbol, dvar::Any}[]
    equations_junk = Expr(:block)
    
    for item ∈ args[1].args
        if item isa LineNumberNode
            push!(equations_inner_lnns, item)
        elseif isexpr(item, :(=))
            lhs, rhs = item.args
            if isexpr(lhs, :call, 2) && lhs.args[1] == :D && lhs.args[2] isa Symbol
                var = lhs.args[2]
                push!(equations_parsed, (; var, dvar=rhs))
            else
                error(ArgumentError("Malformed entry in @equations, each entry must be of the form `D(x) = expr`, got \n$item"))
            end
        elseif isexpr(item, :macrocall) && item.args[1] == Symbol("@setup")
            push!(equations_junk.args, item.args[2:end]...)
        else
            error(ArgumentError("Malformed entry in @equations, each entry must be of the form `D(x) = expr`, got \n$item"))
        end 
    end
    
    equations_syms = [var for (; var) ∈ equations_parsed]
    if equations_syms != state_syms
        error(ArgumentError("Provided differential equations do not match provided states, got differential equations for $equations_syms, and the provided states were $state_syms"))
    end
    
    (equations_parsed, equations_inner_lnns, equations_junk)
end

function parse_noise_equations(args, param_syms, state_syms, input_data)
    if !(@isdefined(param_syms)) && !(@isdefined(state_syms)) && !(@isdefined(input_data))
        error(ArgumentError("@noise_equations must be declared after @parameters, @states, and @inputs"))
    end
    
    if length(args) > 1
        error(ArgumentError("@noise_equations expects a single begin...end block"))
    end
    if !isexpr(args[1], :block)
        error(ArgumentError("@noise_equations expects a begin...end block"))
    end
    
    noise_eqs_inner_lnns = LineNumberNode[]
    noise_eqs_parsed = @NamedTuple{var::Symbol, Wvar::Any}[]
    
    for item ∈ args[1].args
        if item isa LineNumberNode
            push!(noise_eqs_inner_lnns, item)
        elseif isexpr(item, :(=))
            lhs, rhs = item.args
            if isexpr(lhs, :call, 2) && lhs.args[1] == :W && lhs.args[2] isa Symbol
                var = lhs.args[2]
                push!(noise_eqs_parsed, (; var, Wvar=rhs))
            else
                error(ArgumentError("Malformed entry in @noise_equations, each entry must be of the form `W(x) = expr`, got \n$item"))
            end
        elseif isexpr(item, :macrocall) && item.args[1] == Symbol("@setup")
            # Could be added to equations_junk if needed
        else
            error(ArgumentError("Malformed entry in @noise_equations, each entry must be of the form `W(x) = expr`, got \n$item"))
        end 
    end
    
    noise_eqs_syms = [var for (; var) ∈ noise_eqs_parsed]
    for sym ∈ noise_eqs_syms
        if sym ∉ state_syms
            error(ArgumentError("Got a noise equation that does not match provided states, got noise equations $noise_eqs_syms, and the provided states were $state_syms"))
        end
    end
    
    (noise_eqs_parsed, noise_eqs_inner_lnns)
end

function parse_discrete_events(args)
    if length(args) > 1
        error(ArgumentError("@discrete_events currently only supports single events"))
    end
    
    ex = only(args)
    if !isexpr(ex, :call) || ex.args[1] != :(=>)
        error("@discrete events requires inputs of the form condition => affects, got $ex")
    end
    
    disc_cond, disc_affect = ex.args[2:end]
    disc_affects = []
    
    if isexpr(disc_affect, :(=))
        push!(disc_affects, (var=disc_affect.args[1], val=disc_affect.args[2]))
    elseif isexpr(disc_affect, :tuple)
        for arg ∈ disc_affect.args
            if isexpr(arg, :parameters)
                for kwarg ∈ arg.args
                    if isexpr(kwarg, :kw)
                        push!(disc_affects, (var=kwarg.args[1], val=kwarg.args[2]))
                    else
                        error("Malformed affect $kwarg")
                    end
                end
            else
                if isexpr(arg, :(=))
                    push!(disc_affects, (var=arg.args[1], val=arg.args[2]))
                else
                    error("Malformed affect $arg")
                end
            end
        end
    else
        error("@discrete_event expected an affect argument of the form sym=val or (sym1=val1, sym2=val2, ...), got $disc_affect")
    end
    
    (disc_cond, disc_affects)
end

function parse_continuous_events(args)
    if length(args) > 1
        error(ArgumentError("@continuous_events currently only supports single events"))
    end
    
    ex = only(args)
    if !isexpr(ex, :call) || ex.args[1] != :(=>)
        error("@continuous events requires inputs of the form condition => affects, got $ex")
    end
    
    cont_cond, cont_affect = ex.args[2:end]
    cont_affects = []
    
    if isexpr(cont_affect, :(=))
        push!(cont_affects, (var=cont_affect.args[1], val=cont_affect.args[2]))
    elseif isexpr(cont_affect, :tuple)
        for arg ∈ cont_affect.args
            if isexpr(arg, :parameters)
                for kwarg ∈ arg.args
                    if isexpr(kwarg, :kw)
                        push!(cont_affects, (var=kwarg.args[1], val=kwarg.args[2]))
                    else
                        error("Malformed affect $kwarg")
                    end
                end
            else
                if isexpr(arg, :(=))
                    push!(cont_affects, (var=arg.args[1], val=arg.args[2]))
                else
                    error("Malformed affect $arg")
                end
            end
        end
    else
        error("@discrete_event expected an affect argument of the form sym=val or (sym1=val1, sym2=val2, ...), got $disc_affect")
    end
    
    (cont_cond, cont_affects)
end

function parse_computed_properties(lnn, item_args, param_syms, state_syms)
    computed_prop_data = @NamedTuple{prop::Symbol, body::Any, lnn::Union{Nothing, LineNumberNode}}[]
    for arg ∈ item_args
        if isexpr(arg, :(=), 2)
            push!(computed_prop_data, (prop=arg.args[1], body=arg.args[2], lnn=lnn))
        elseif isexpr(arg, :block)
            for arg_arg ∈ arg.args
                if arg_arg isa LineNumberNode
                    lnn = arg_arg
                elseif isexpr(arg_arg, :(=), 2)
                    push!(computed_prop_data, (prop=arg_arg.args[1], body=arg_arg.args[2], lnn=lnn))
                else
                    error("Unrecognized computed property. Computed properties must be of the form prop = expr, got $arg_arg")
                end
            end
        else
            error("Unrecognized computed property. Computed properties must be of the form prop = expr, or a begin/end block of such declarations. Got $arg")
        end
    end
    computed_prop_data
end

function emit_struct_definition(blox_name, suptype, sig, param_syms, extra_field_data, 
                                param_vals_ex, constructor_junk, __source__)
    esc_blox_name = esc(blox_name)
    maybe_remove_type(ex) = isexpr(ex, :(::)) ? ex.args[1] : ex
    
    output_constructor = Expr(:function, esc(sig), quote
        $(esc(constructor_junk))
        $(esc(:param_vals)) = $(esc(param_vals_ex))
        $(Expr(:block, (:($(esc(maybe_remove_type(field_name))) = $(esc(field_val)))
                        for (;field_name, field_val) ∈ extra_field_data)...))
        new(
            $(esc(:name)),
            $(esc(:namespace)),
            $(esc(:param_vals)),
            $((esc(maybe_remove_type(field_name)) for (;field_name) ∈ extra_field_data)...)
        )
    end)
    set_fn_lnn!(output_constructor, __source__)
    
    output_struct_def = :(Core.@__doc__ struct $esc_blox_name <: $(esc(suptype))
        name::Symbol
        namespace::Union{Symbol, Nothing}
        param_vals::@NamedTuple{$(param_syms...)}
        $((esc(field_name) for (;field_name) ∈ extra_field_data)...)
        $output_constructor
    end)
    
    def_nameof = :(function Base.nameof(x::$esc_blox_name)
        getfield(x, :name)
    end)
    Expr(:block, output_struct_def, def_nameof)
end

function emit_getproperty(blox_name, __source__)
    esc_blox_name = esc(blox_name)
    def_getprop = :(function Base.getproperty(blox::$esc_blox_name, prop::Symbol)
        fwd_getproperty(blox, prop)
    end)
    set_fn_lnn!(def_getprop, __source__)
    def_getprop
end

function emit_param_symbols(blox_name, param_syms, params_lnn)
    esc_blox_name = esc(blox_name)
    def_param_symbols = :(function NeurobloxBase.param_symbols(::Type{$esc_blox_name})
        $(Expr(:tuple, (QuoteNode(sym) for sym in param_syms)...))
    end)
    set_fn_lnn!(def_param_symbols, params_lnn)
    def_param_symbols
end

function emit_state_symbols(blox_name, state_syms, states_lnn)
    esc_blox_name = esc(blox_name)
    def_state_symbols = :(function NeurobloxBase.state_symbols(::Type{$esc_blox_name})
        $(Expr(:tuple, (QuoteNode(sym) for sym ∈ state_syms)...))
    end)
    set_fn_lnn!(def_state_symbols, states_lnn)
    def_state_symbols
end

function emit_input_symbols(blox_name, inputs_data, inputs_lnn)
    esc_blox_name = esc(blox_name)
    
    def_input_symbols = :(function NeurobloxBase.input_symbols(::Type{$esc_blox_name})
        $(Expr(:tuple, (QuoteNode(sym) for (sym, val) in inputs_data)...))
    end)
    set_fn_lnn!(def_input_symbols, inputs_lnn)
    
    def_gdy_inputs = :(function GraphDynamics.initialize_input(sys::Subsystem{$esc_blox_name})
        $(Expr(:tuple,
               Expr(:parameters,
                    (Expr(:kw, esc(var), esc(val)) for (var, val) ∈ inputs_data)...)))
    end)
    set_fn_lnn!(def_gdy_inputs, inputs_lnn)
    
    Expr(:block, def_input_symbols, def_gdy_inputs)
end

function emit_output_symbols(blox_name, outputs_syms)
    esc_blox_name = esc(blox_name)
    
    def_output_symbols = :(function NeurobloxBase.output_symbols(::Type{$esc_blox_name})
        $(Expr(:tuple, (QuoteNode(sym) for sym ∈ outputs_syms)...))
    end)
    
    def_outputs = :(function NeurobloxBase.outputs(blox::Subsystem{$esc_blox_name})
        $(Expr(:tuple,
               Expr(:parameters,
                    (Expr(:kw, name,
                          Expr(:call, getproperty, :blox, QuoteNode(name)))
                     for name ∈ outputs_syms)...)))
    end)
    
    Expr(:block, def_output_symbols, def_outputs)
end

function emit_event_times(blox_name, param_syms, state_syms, ev_time_arg)
    esc_blox_name = esc(blox_name)
    def_ev_time = :(function GraphDynamics.event_times(blox::Subsystem{$esc_blox_name})
        ((; $(esc.(param_syms)...),) = blox)
        ((; $(esc.(state_syms)...),) = blox)
        $(esc(ev_time_arg))
    end)
    def_ev_time
end

function emit_discrete_events(blox_name, param_syms, state_syms, disc_cond, disc_affects)
    esc_blox_name = esc(blox_name)
    
    set_ex = Expr(:block)
    for (var, ex) ∈ disc_affects
        push!(set_ex.args, quote
                  $(esc(var)) = $(esc(ex))
                  sys_view.$(var)[] = $(esc(var))
              end)
    end
    
    def_disc_ev = quote
        GraphDynamics.has_discrete_events(::Type{$esc_blox_name}) = true
        
        function GraphDynamics.discrete_event_condition(sys::Subsystem{$esc_blox_name}, $(esc(:t)), _)
            ((; $(esc.(param_syms)...),) = sys)
            ((; $(esc.(state_syms)...),) = sys)
            $(esc(disc_cond))
        end
        
        function GraphDynamics.apply_discrete_event!(integrator, sys_view, sys::Subsystem{$esc_blox_name}, _)
            ((; $(esc.(param_syms)...),) = sys)
            ((; $(esc.(state_syms)...),) = sys)
            $set_ex
            nothing
        end
    end
    def_disc_ev
end

function emit_continuous_events(blox_name, param_syms, state_syms, cont_cond, cont_affects)
    esc_blox_name = esc(blox_name)
    
    set_ex = Expr(:block)
    for (var, ex) ∈ cont_affects
        push!(set_ex.args, quote
                  $(esc(var)) = $(esc(ex))
                  sys_view.$(var)[] = $(esc(var))
              end)
    end
    
    def_cont_ev = quote
        GraphDynamics.has_continuous_events(::Type{$esc_blox_name}) = true
        
        function GraphDynamics.continuous_event_condition(sys::Subsystem{$esc_blox_name}, $(esc(:t)), _)
            ((; $(esc.(param_syms)...),) = sys)
            ((; $(esc.(state_syms)...),) = sys)
            $(esc(cont_cond))
        end
        
        function GraphDynamics.apply_continuous_event!(integrator, sys_view, sys::Subsystem{$esc_blox_name}, _)
            ((; $(esc.(param_syms)...),) = sys)
            ((; $(esc.(state_syms)...),) = sys)
            $set_ex
            nothing
        end
    end
    def_cont_ev
end

function emit_subsystem_methods(blox_name, param_syms, state_data, state_syms, 
                                input_syms, equations_parsed, equations_junk,
                                should_emit_equations,
                                params_lnn, states_lnn, __source__)
    esc_blox_name = esc(blox_name)
    
    def_to_subsystem = Expr(:function, 
        :(GraphDynamics.to_subsystem(blox::$esc_blox_name)),
        Expr(:block,
            __source__,
            params_lnn,
            :(($(esc.(param_syms)...),) = getfield(blox, :param_vals)),
            states_lnn,
            (:($(esc(sym)) = $(esc(val))) for (sym, val) ∈ state_data)...,
            quote
                params = SubsystemParams{$esc_blox_name}($(Expr(:tuple,
                    Expr(:parameters, esc.(param_syms)...))))
                states = SubsystemStates{$esc_blox_name}($(Expr(:tuple,
                    Expr(:parameters, esc.(state_syms)...))))
                Subsystem(states, params)
            end
        )
    )
    def_subsystem_differential = if should_emit_equations
        Expr(:function, 
             :(GraphDynamics.subsystem_differential(sys::Subsystem{$esc_blox_name}, (; $(esc.(input_syms)...)), $(esc(:t)))),
             Expr(:block,
                  __source__,
                  :($(esc(:__sys__)) = sys),
                  params_lnn,
                  :(((; $(esc.(param_syms)...),) = sys)),
                  states_lnn,
                  :(((; $(esc.(state_syms)...),) = sys)),
                  esc(equations_junk),
                  Expr(:block, (Expr(:(=), Symbol(:D, var), esc(dvar))
                                for (; var, dvar) ∈ equations_parsed)...),
                  :(SubsystemStates{$esc_blox_name}(; $((:($(esc(var)) = $(Symbol(:D, var))) 
                                                         for (;var) ∈ equations_parsed)...)))
                  )
             )
    end
    Expr(:block, def_to_subsystem, def_subsystem_differential)
end

function emit_stochastic_methods(blox_name, param_syms, state_syms, noise_eqs_parsed, noise_eqs_inner_lnns)
    esc_blox_name = esc(blox_name)
    
    noise_body = let ex = Expr(:block)
        for ((; var, Wvar), lnn) ∈ zip(noise_eqs_parsed, noise_eqs_inner_lnns)
            i = findfirst(==(var), state_syms)
            @assert !isnothing(i)
            push!(ex.args, lnn, :(@inbounds v.$(state_syms[i])[] = $(esc(Wvar))))
        end
        ex
    end
    
    def_stoch = quote
        GraphDynamics.isstochastic(::Type{$esc_blox_name}) = true
        
        function GraphDynamics.apply_subsystem_noise!(v, sys::Subsystem{$esc_blox_name}, $(esc(:t)))
            ((; $(esc.(param_syms)...),) = sys)
            ((; $(esc.(state_syms)...),) = sys)
            $noise_body
        end
    end
    
    def_stoch
end


function emit_computed_properties(blox_name, param_syms, state_syms, comp_prop_data)
    esc_blox_name = esc(blox_name)
    :(function GraphDynamics.computed_properties(::Type{$esc_blox_name})
          $(let defs = Expr(:block)
                for (; prop, body, lnn) ∈ comp_prop_data
                    fdef = :(function $prop(sys)
                                 ((; $(esc.(param_syms)...),) = sys)
                                 ((; $(esc.(state_syms)...),) = sys)
                                 $(esc(body))
                             end)
                    push!(defs.args, lnn, fdef)                    
                end
                defs
            end)
          $(Expr(:tuple, (:($prop = $prop) for (;prop) ∈ comp_prop_data)...))
      end)
end


function emit_computed_properties_with_inputs(blox_name, param_syms, state_syms, input_syms, comp_prop_data)
    esc_blox_name = esc(blox_name)
    :(function GraphDynamics.computed_properties_with_inputs(::Type{$esc_blox_name})
          $(let defs = Expr(:block)
                for (; prop, body, lnn) ∈ comp_prop_data
                    fdef = :(function $prop(sys, input)
                                 ((; $(esc.(param_syms)...),) = sys)
                                 ((; $(esc.(state_syms)...),) = sys)
                                 ((; $(esc.(input_syms)...),) = input)
                                 $(esc(body))
                             end)
                    push!(defs.args, lnn, fdef)                    
                end
                defs
            end)
          $(Expr(:tuple, (:($prop = $prop) for (;prop) ∈ comp_prop_data)...))
      end)
end
