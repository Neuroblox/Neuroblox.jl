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
    "./docs/src/tutorials/spectralDCM.jl"
    ],
    "./docs/src/tutorials";
    documenter = true
)

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(Neuroblox, :DocTestSetup, :(using Neuroblox); recursive = true)

include("pages.jl")

makedocs(sitename = "Neuroblox",
    authors = "Neuroblox Inc.",
    modules = [Neuroblox],
    clean = true, doctest = false, linkcheck = false,
    warnonly = [:docs_block, :missing_docs, :linkcheck],
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
        #canonical = "https://docs.sciml.ai/LinearSolve/stable/"),
    pages = pages)

repo =  "github.com/Neuroblox/NeurobloxDocsHost"

withenv("GITHUB_REPOSITORY" => repo) do
    deploydocs(; repo = repo, push_preview = true)
end
