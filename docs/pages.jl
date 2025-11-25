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
    "API" => ["api.md", "blox_api.md"],
    "Neuroblox Course" => Any[
        "course_tutorials/intro_diffeq.md",
        "course_tutorials/intro_plot.md",
        "course_tutorials/blox_connections.md",
        "course_tutorials/neuron_mass.md",
        "course_tutorials/PING_circuit.md",
        "course_tutorials/CS_circuit.md",
        "course_tutorials/decision_making.md",
        "course_tutorials/learning.md",
        #"course_tutorials/optimization.md",
    ],
    "Release Notes" => "release_notes.md",
]
