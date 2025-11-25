using Neuroblox
using Documenter
using Literate

Literate.markdown("./docs/src/getting_started.jl", "./docs/src/"; documenter = true)

Literate.markdown.([
    "./docs/src/tutorials/resting_state.jl",
    "./docs/src/tutorials/parkinsons.jl",
    "./docs/src/tutorials/neural_assembly.jl",
    "./docs/src/tutorials/ping_network.jl",
    "./docs/src/tutorials/basal_ganglia.jl",
    ],
    "./docs/src/tutorials";
    documenter = true
)

Literate.markdown.([
    "./docs/src/dev_tutorials/graph_dynamics_interop.jl"
],
    "./docs/src/dev_tutorials";
    documenter = true
)
Literate.markdown.([
    "./docs/src/course_tutorials/intro_diffeq.jl",
    "./docs/src/course_tutorials/intro_plot.jl",
    "./docs/src/course_tutorials/blox_connections.jl",
    "./docs/src/course_tutorials/neuron_mass.jl",
    "./docs/src/course_tutorials/PING_circuit.jl",
    "./docs/src/course_tutorials/CS_circuit.jl",
    "./docs/src/course_tutorials/decision_making.jl",
    "./docs/src/course_tutorials/learning.jl",
    #"./docs/src/course_tutorials/optimization.jl",
   ],
    "./docs/src/course_tutorials";
    documenter = true
)


cp("./Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(Neuroblox, :DocTestSetup, :(using Neuroblox); recursive = true)

include("pages.jl")

makedocs(sitename = "Neuroblox Documentation",
    authors = "Neuroblox Inc.",
    modules = [Neuroblox, NeurobloxBase, NeurobloxPharma, NeurobloxBasics, NeurobloxDBS],
    remotes = nothing,
    clean = true, doctest = false, linkcheck = false,
    warnonly = [:docs_block, :missing_docs, :linkcheck],
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
    pages = pages)

repo =  "github.com/Neuroblox/NeurobloxDocsHost"

withenv("GITHUB_REPOSITORY" => repo) do
    deploydocs(; repo = repo, push_preview = true)
end
