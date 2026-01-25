# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "How to install" => "install.md",
    "Getting Started" => "getting_started.md",
    "Examples" => Any[
        "examples/resting_state.md",
        "examples/parkinsons.md",
        "examples/basal_ganglia.md",
        "examples/neural_assembly.md",
    ],
    "Tutorials" => Any[
        # "tutorials/intro_diffeq.md",
        # "tutorials/intro_plot.md",
        # "tutorials/blox_connections.md",
        # "tutorials/neuron_mass.md",
        "tutorials/receptors.md",
        "tutorials/PING_circuit.md",
        "tutorials/CS_circuit.md",
        "tutorials/decision_making.md",
        "tutorials/learning.md",
        #"tutorials/optimization.md",
    ],
    "API" => ["api_graph.md", "api_blox.md", "api_receptors.md", "api_plot.md", "api_utils.md", "api_rl.md", "at_blox_macro.md", "experiment_macro.md", "connection_macro.md"],
    "Release Notes" => "release_notes.md",
]
