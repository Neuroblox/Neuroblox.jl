using SafeTestsets

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end
@time @safetestset "ODE from Graph Tests" begin include("ode_from_graph.jl") end
@time @safetestset "Neural Signal Measurement Models Tests" begin include("measurement_models.jl") end
@time @safetestset "Spectral Utilities Tests" begin include("spectraltools.jl") end
@time @safetestset "Data Fitting Tests" begin include("datafitting.jl") end
@time @safetestset "ODE from Graph and simulate" begin include("graph_to_dataframe.jl") end
@time @safetestset "Learning Tests" begin include("learning.jl") end
@time @safetestset "Control Tests" begin include("controllers.jl") end
@time @safetestset "Source Tests" begin include("source_components.jl") end
# fitting tests should be at the end since they take the longest
# removing them for now until we have real fitting tests
# @time @safetestset "Fitting Tests" begin include("fitting.jl") end
# @time @safetestset "Bayesian Fitting Tests" begin include("bayesian_fitting.jl") end


