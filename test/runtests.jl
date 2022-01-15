using SafeTestsets

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Fitting Tests" begin include("fitting.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end
@time @safetestset "ODE from Graph Tests" begin include("ode_from_graph.jl") end
@time @safetestset "Spectral Tests" begin include("spectrum.jl") end
@time @safetestset "Spectral Utilities Tests" begin include("spectraltools.jl") end
