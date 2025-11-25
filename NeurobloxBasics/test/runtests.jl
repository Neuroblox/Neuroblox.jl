using SafeTestsets

@safetestset "Aqua" begin 
    using Aqua, NeurobloxBasics
    Aqua.test_all(NeurobloxBasics; persistent_tasks = false)
end

@time @safetestset "Utilities" begin include("utils.jl") end
@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end

@time @safetestset "GraphDynamics vs MTK tests" begin include("GraphDynamicsTests/runtests.jl") end
