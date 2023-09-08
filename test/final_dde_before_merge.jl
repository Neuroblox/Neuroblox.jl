# Intro setup
using ModelingToolkit, DelayDiffEq, Graphs, MetaGraphs, Plots
using ModelingToolkit: get_namespace, get_systems, renamespace, namespace_equation, namespace_expr
import ModelingToolkit: inputs, outputs, nameof
@variables t
D = Differential(t)

# blox_utilities.jl
namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(nothing, name) = Symbol(name)
namespaceof(blox) = blox.namespace
get_sys(blox) = blox.odesystem

function scope_dict(para_dict::Dict{Symbol, Union{Real,Num}})
    para_dict_copy = copy(para_dict)
    for (n,v) in para_dict_copy
        if typeof(v) == Num
            para_dict_copy[n] = ParentScope(v)
        else
            para_dict_copy[n] = (@parameters $n=v)[1]
        end
    end
    return para_dict_copy
end

# REDEFINED
nameof(blox) = (nameof ∘ get_sys)(blox) # Redefined because no odesys to get name from

function get_namespaced_sys(blox)
    sys = get_sys(blox)
    ODESystem(
        equations(sys), 
        independent_variable(sys), 
        states(sys), 
        parameters(sys); 
        name = namespaced_name(inner_namespaceof(blox), nameof(blox))
    ) 
end

function inner_namespaceof(blox)
    parts = split((string ∘ namespaceof)(blox), '₊')
    if length(parts) == 1
        return nothing
    else
        return join(parts[2:end], '₊')
    end
end

function find_eq(eqs::AbstractVector{<:Equation}, lhs)
    findfirst(eqs) do eq
        lhs_vars = get_variables(eq.lhs)
        length(lhs_vars) == 1 && isequal(only(lhs_vars), lhs)
    end
end

# redefined to be used for delays
function input_equations(blox)
    sys = get_sys(blox)
    inps = inputs(sys)
    sys_eqs = equations(sys)
    # CHANGE HERE
    @variables t
    eqs = map(inps) do inp
        idx = find_eq(sys_eqs, inp)
        if isnothing(idx)
            namespace_equation(
                inp ~ 0, 
                nothing, 
                namespaced_name(inner_namespaceof(blox), nameof(blox));
                ivs = t
            )
        else
            namespace_equation(
                sys_eqs[idx], 
                nothing, 
                namespaced_name(inner_namespaceof(blox), nameof(blox));
                ivs = t
            )
        end
    end

    return eqs
end

# Neurographs.jl
function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

weight_parameters(blox) = Num[]

# connections.jl
mutable struct BloxConnector
    eqs::Vector{Equation}
    params::Vector{Num}

    BloxConnector() = new(Equation[], Num[])

    function BloxConnector(bloxs)
        eqs = reduce(vcat, input_equations.(bloxs)) 
        params = reduce(vcat, weight_parameters.(bloxs))
        #eqs = namespace_equation.(eqs, nothing, namespace)
        new(eqs, params)
    end
end

function (bc::BloxConnector)(
    jc::JansenRitCBloxDelay, 
    bloxin; 
    weight = 1,
    delay = 0
)
    # Need t for the delay term
    @variables t

    sys_out = get_namespaced_sys(jc)
    sys_in = get_namespaced_sys(bloxin)

    # Define & accumulate delay parameter
    τ_name = Symbol("τ_$(nameof(sys_out))_$(nameof(sys_in))")
    τ = only(@parameters $(τ_name)=delay)
    push!(bc.params, τ)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    w = only(@parameters $(w_name)=weight)
    push!(bc.params, w)

    x = namespace_expr(jc.connector, nothing, nameof(sys_out))
    eq = sys_in.jcn ~ x(t-τ)*w
    
    accumulate_equation!(bc, eq)
end

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs +  eq.rhs

end

# Neurographs.jl
function get_sys(g::MetaDiGraph)
    map(vertices(g)) do v
        b = get_prop(g, v, :blox)
        get_sys(b)
    end
end

function system_from_graph(g::MetaDiGraph; name)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc; name)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector; name)
    @variables t
    blox_syss = get_sys(g)
    return compose(ODESystem(bc.eqs, t, [], bc.params; name), blox_syss)
end

function get_blox(g::MetaDiGraph)
    map(vertices(g)) do v
        get_prop(g, v, :blox)
    end
end

# switch to system_from_graph
function connector_from_graph(g::MetaDiGraph)
    bloxs = get_blox(g)
    link = BloxConnector(bloxs)
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        for vn in inneighbors(g, v)
            bn = get_prop(g, vn, :blox)
            w = get_prop(g, vn, v, :weight)
            d = get_prop(g, vn, v, :delay) # CHANGE HERE
            link(bn, b; weight = w, delay = d)
        end
    end

    return link
end 


### MERGED SO FAR ###


# neural_mass.jl
mutable struct JansenRitCBloxDelay
    p_dict::Dict{Symbol,Union{Real,Num}}
    eqs::Vector{Equation}
    sts::Vector{Any}
    connector
    jcn
    odesystem
    namespace
    function JansenRitCBloxDelay(;name, τ=0.001, H=20.0, λ=5.0, r=0.15)
        para_dict = scope_dict(Dict{Symbol,Union{Real,Num}}(:τ => τ, :H => H, :λ => λ, :r => r))
        τ=para_dict[:τ]
        H=para_dict[:H]
        λ=para_dict[:λ]
        r=para_dict[:r]
        sts = @variables x(..)=1.0 y(t)=1.0 jcn(t)=0.0 [input=true]
        eqs = [D(x(t)) ~ y - ((2/τ)*x(t)),
               D(y) ~ -x(t)/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
        odesystem = System(eqs, name=name)
        new(para_dict, eqs, sts, sts[1], sts[3], odesystem, nothing)
    end
end



# test blox
@named PY  = JansenRitCBloxDelay(τ=0.001, H=20, λ=5, r=0.15)
@named EI  = JansenRitCBloxDelay(τ=0.01, H=20, λ=5, r=5)
@named II  = JansenRitCBloxDelay(τ=2.0, H=60, λ=5, r=5)

# test graphs
g = MetaDiGraph()
add_blox!(g, PY)
add_blox!(g, EI)
add_blox!(g, II)
add_edge!(g, 1, 2, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 2, 3, Dict(:weight => 1.0, :delay => 0.5))
add_edge!(g, 3, 1, Dict(:weight => 1.0, :delay => 1.5))

@named final_system = system_from_graph(g)
sim_dur = 10.0 # Simulate for 10 Seconds
sys = structural_simplify(final_system)
prob = DDEProblem(sys,
    [],
    (0.0, sim_dur),
    constant_lags = [1])
alg = MethodOfSteps(Tsit5())
sol_mtk = solve(prob, alg, reltol = 1e-7, abstol = 1e-10, saveat=0.001)

# Notes for 9/6 meeting

# Review changes (namespacing with iv = t)
# Review changes (delay)
# Multiple dispatch or separate bloxs for delay?
# Where to collect lags?