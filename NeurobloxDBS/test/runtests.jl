using SafeTestsets
@safetestset "Aqua" begin 
    using Aqua, NeurobloxDBS
    Aqua.test_all(NeurobloxDBS; persistent_tasks = false)
end

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "DBS" begin include("dbs.jl") end

@time @safetestset "GraphDynamics vs MTK tests" begin include("GraphDynamicsTests/runtests.jl") end
