using SafeTestsets

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Fitting Tests" begin include("fitting.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end
