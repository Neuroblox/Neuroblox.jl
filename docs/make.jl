using Neuroblox
using Documenter
using Literate

# HACK to get previews working
Documenter.verify_github_pull_repository(repo, prnr) = true

Literate.markdown("./docs/src/getting_started.jl", "./docs/src/"; documenter = true)

Literate.markdown.([
    "./docs/src/examples/resting_state.jl",
    "./docs/src/examples/parkinsons.jl",
    "./docs/src/examples/neural_assembly.jl",
    "./docs/src/examples/basal_ganglia.jl",
    ],
    "./docs/src/examples";
    documenter = true
)

Literate.markdown.([
    # "./docs/src/tutorials/intro_diffeq.jl",
    # "./docs/src/tutorials/intro_plot.jl",
    # "./docs/src/tutorials/blox_connections.jl",
    # "./docs/src/tutorials/neuron_mass.jl",
    "./docs/src/tutorials/receptors.jl",
    "./docs/src/tutorials/PING_circuit.jl",
    "./docs/src/tutorials/CS_circuit.jl",
    "./docs/src/tutorials/decision_making.jl",
    "./docs/src/tutorials/learning.jl",
    #"./docs/src/tutorials/optimization.jl",
   ],
    "./docs/src/tutorials";
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
