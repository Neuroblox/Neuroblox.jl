using SafeTestsets

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Fitting Tests" begin include("fitting.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end
@time @safetestset "ODE from Graph Tests" begin include("ode_from_graph.jl") end
@time @safetestset "Neural Signal Measurement Models Tests" begin include("measurement_models.jl") end
@time @safetestset "Spectral Utilities Tests" begin include("spectral_tools.jl") end
@time @safetestset "Dynamic Causal Modeling Tests" begin include("DCM.jl") end
@time @safetestset "ODE from Graph and simulate" begin include("graph_to_dataframe.jl") end
