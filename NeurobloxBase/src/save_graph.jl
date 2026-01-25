"""
    save_graph(path::String, g::GraphSystem; save_manifest = true)

Save a GraphSystem as a jld2 file. If `save_manifest` is set, then the current Manifest.toml of the active project will also be saved as a string.
"""
function save_graph(path::String, g::GraphSystem; save_manifest = true)
    @assert endswith(path, ".jld2") "Must provide a .jld2 file to save the graph."

    data_dict = gs_jld2_dict(g)

    if save_manifest
        manifest_path = joinpath(dirname(Base.active_project()), "Manifest.toml")
        if isfile(manifest_path)
            JLD2.save(path, merge(data_dict, Dict("manifest" => read(manifest_path, String))))
        else
            @warn "No manifest found at $manifest_path, will not be saved."
            JLD2.save(path, data_dict)
        end
    else
        JLD2.save(path, data_dict)
    end
end

function gs_jld2_dict(g::GraphSystem)
    g_name = g.name
    g_data = zip(keys(g.data), collect.(values(g.data)))
    fg = g.flat_graph
    Dict("data" => (g_name, g_data, fg))
end

"""
    load_graph(path::String)

Load a GraphSystem from a saved GraphSystem or ODEProblem as a jld2 file.
"""
function load_graph(path::String; from_dsl = false)
    @assert endswith(path, ".jld2") "Must provide a .jld2 file from which to load the graph."
    (g_name, g_data, fg) = JLD2.load(path)["data"]

    g_data = OrderedDict(map(g_data) do (k, v)
                             k => OrderedDict(v)
                         end)
    return GraphSystem(g_name, OrderedDict(g_data), fg)
end

"""
    save_problem(path::String, prob::ODEProblem)

Save an ODEProblem as a jld2 file. This saves the graph, initial condition, tspan, and parameters.
"""
function save_problem(path::String, prob::ODEProblem)
    @assert endswith(path, ".jld2") "Must provide a .jld2 file to save the problem."

    pmap = map(parameter_symbols(prob)) do sym
        sym => prob.ps[sym]
    end
    u0map = map(enumerate(variable_symbols(prob))) do (i, sym)
        sym => prob.u0[i]
    end

    g_dict = gs_jld2_dict(prob.p.graph)

    JLD2.save(path, merge(g_dict, Dict("pmap" => pmap, "u0map" => u0map, "tspan" => prob.tspan)))
end

"""
    load_problem(path::String)

Load an ODEProblem from a .jld2 file.
"""
function load_problem(path::String)
    g = load_graph(path)
    pmap = JLD2.load(path)["pmap"]
    u0map = JLD2.load(path)["u0map"]
    tspan = JLD2.load(path)["tspan"]

    ODEProblem(g, u0map, tspan, pmap)
end

"""
    @save_graph "file_name.jld2" @graph g begin
        ...
    end

Insert this macrocall before the DSL definition of a graph to save both the graph, as well as the DSL code used to generate it. This is useful to ensure that the model is reproducible between versions.

IMPORTANT: the reconstructed graph may have slightly different connections if random numbers are used when flattening a composite (like a Cortical blox). Do not rely on the topology being the exact same.
"""
macro save_graph(path::String, expr)
    @assert endswith(path, ".jld2") "Must specify a .jld2 file to save the graph in."

    if (Base.isexpr(expr, :macrocall) && (expr.args[1] === Symbol("@graph")))
        @gensym g

        dsl_expr = Meta.quot(Base.remove_linenums!(expr))
        manifest_path = joinpath(@__DIR__, "../../Manifest.toml")

        quote
            $g = $(esc(expr))
            $JLD2.save($path, $Dict("g" => $g, "manifest" => $read($manifest_path, $String), "dsl" => $string($dsl_expr)))
            $g
        end
    else
        :(save_graph($path, $(esc(expr))); $(esc(expr)))
    end
end
