using SafeTestsets

@safetestset "Aqua" begin 
    using Aqua, NeurobloxBase
    Aqua.test_all(NeurobloxBase; persistent_tasks = false)
end
@safetestset "DSL" begin include("dsl.jl") end
