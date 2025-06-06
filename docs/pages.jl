# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "How to install" => "install.md",
    "Getting Started" => "getting_started.md",
    "Tutorials" => Any[
        "tutorials/resting_state.md",
        "tutorials/ping_network.md",
        "tutorials/parkinsons.md",
        "tutorials/basal_ganglia.md",
        "tutorials/neural_assembly.md",
    ],
    "Dev Tutorials" => ["dev_tutorials/graph_dynamics_interop.md"],
    "API" => "api.md",
    "Release Notes" => "release_notes.md",
]
