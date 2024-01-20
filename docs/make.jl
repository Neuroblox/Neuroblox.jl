using Neuroblox
using Documenter

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(Neuroblox, :DocTestSetup, :(using Neuroblox); recursive = true)

include("pages.jl")

makedocs(sitename = "Neuroblox",
    authors = "Neuroblox Inc.",
    modules = [Neuroblox],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        #canonical = "https://docs.sciml.ai/LinearSolve/stable/"),
    pages = pages)

deploydocs(;
    repo = "github.com/Neuroblox/Neuroblox.jl",
    push_preview = true)
