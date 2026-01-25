"""
g = @graph begin
    ...
end
prob = ODEProblem(g, ...)

prob_anes = @experiment prob begin
    @setup begin
        p_ts = 2.5
        p_g = 2.5
    end

    GABA_A_Synapse,
        τ₂ -> τ₂ * p_ts

    HHNeuronInhib,
        G_syn -> G_syn * p_g

    (HHNeuronInhib ∈ (Nrt_A, Nrt_B)),
        I_bg -> 3 - 3 * p_g

    (HHNeuronExci ∈ (Thal_core_A, Thal_core_B)),
        I_bg -> (I_bg - 0.25) * p_g + 0.25
end

Macro that provides a generic interface for modifying initial conditions, time span, and parameters of an ODEProblem.

New initial conditions and time span can be declared using the @initial_conditions and @tspan macros, respectively.

Auxiliary variables can be declared inside a @setup block.

Other lines are understood as parameter changes. Each line is a tuple that consists of the following:

1. The blox that will be affected. This can be one of the following:
    - A single blox (e.g. hh_neuron1)
    - All blox of a given type (e.g. HHNeuronInhib)
    - All blox of a given type in a given composite (e.g. HHNeuronInhib ∈ thal1)
    - All synapses of a given type going between two regions (e.g. GABA_A_Synapse ∈ (cort1 => cort2))

No other specification is recognized.

2. The rule to update a parameter of the selected blox. This is specified as an anonymous function, where the name of the argument is the parameter to be affected.
"""
macro experiment(probsym, expr)
    output = Expr(:block)
    esc_probsym = esc(probsym)
    @gensym new_param_obj new_prob

    remake_args = :((; p = $new_param_obj))

    for line in expr.args
        if Base.isexpr(line, :macrocall)
            line_name, line_lnn, line_args = line.args

            if line_name == Symbol("@initial_conditions")
                push!(remake_args.args[1].args, Expr(:kw, :u0, esc(line_args)))
            elseif line_name == Symbol("@tspan")
                push!(remake_args.args[1].args, Expr(:kw, :tspan, esc(line_args)))
            elseif line_name == Symbol("@setup")
                for arg in line_args.args
                    push!(output.args, esc(arg))
                end
            else
                error("Unrecognized macro call $line in the @experiment macro. Use @initial_conditions to declare new initial conditions and @tspan to declare a new tspan.")
            end
        elseif Base.isexpr(line, :tuple)
            if length(line.args) != 2
                error("Malformed line: \n$line\nThis expects a blox specification, and a parameter update in anonymous function syntax, separated by a comma.")
            elseif line.args[2].head != :(->)
                error("Parameter update must be written using anonymous function syntax, with a ->.")
            end

            push!(output.args, emit_setp_code(line, new_param_obj))
        elseif line isa LineNumberNode
            continue
        else
            error("Unrecognized input line. Valid input lines are @dose_parameters declarations and effect lines.")
        end
    end
    pushfirst!(output.args, :($new_param_obj = $copy($esc_probsym.p)))
    push!(output.args, :($new_prob = $remake($esc_probsym; $remake_args...)))

    output
end

function emit_setp_code(line, param_obj_sym)
    blox, param_effect = line.args
    param_affected, _ = param_effect.args
    param_name = QuoteNode(param_affected)

    @gensym effect_fn bs syms vals
    
    mod_expr = if blox isa Symbol
        esc_blox = esc(blox)

        quote
            $effect_fn = $(esc(param_effect))
            if $esc_blox isa AbstractBlox
                sym = $Symbol($GraphDynamics.get_name($esc_blox), :₊, $param_name)
                val = $getp($param_obj_sym, sym)($param_obj_sym)
                $param_obj_sym = setp_maybe_typechange($param_obj_sym, sym, val, $effect_fn(val))
            else
                $bs = $Iterators.filter(b -> b isa $esc_blox, $nodes($param_obj_sym.graph.flat_graph))
                $syms = $Symbol[]

                for b in $bs
                    sym = $Symbol($GraphDynamics.get_name(b), :₊, $param_name)
                    push!($syms, sym)
                end
                $vals = $getp($param_obj_sym, $syms)($param_obj_sym)
                $param_obj_sym = setp_maybe_typechange($param_obj_sym, $syms, $vals, $effect_fn.($vals))
            end
        end
    else
        if !(blox.head == :call && (blox.args[1] == :∈ || blox.args[1] == :in))
            error("Use ∈ or in to specify neurons/receptors that reside in (a set of) composite blox.")
        end

        if blox.args[end].head == :call && blox.args[end].args[1] == :(=>)
            error("Modifying parameters for synapses between blox1 => blox2 is currently not supported.")
        end

        neuron_type, composites = blox.args[2:end]
        g = :($mapfoldl(x -> x.graph, $merge!, $(esc(composites)); init=$GraphSystem()))

        quote
            $effect_fn = $(esc(param_effect))
            $syms = $Symbol[]

            for n in $nodes($g)
                if n isa $neuron_type
                    sym = $Symbol($GraphDynamics.get_name(n), :₊, $param_name)
                    push!($syms, sym)
                end
            end
            $vals = $getp($param_obj_sym, $syms)($param_obj_sym)
            $param_obj_sym = setp_maybe_typechange($param_obj_sym, $syms, $vals, $effect_fn.($vals))
        end
    end
end

function setp_maybe_typechange(buffer, syms, vals, new_vals)
    if typeof(vals) !== typeof(new_vals)
        param_map = OrderedDict(zip(syms, new_vals))
        GraphDynamics.set_params!!(buffer, param_map)
    else
        setp(buffer, syms)(buffer, new_vals)
        buffer
    end
end
