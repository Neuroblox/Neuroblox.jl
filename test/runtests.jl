using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Basics"
    @time @safetestset "Utilities" begin include("utils.jl") end
    @time @safetestset "Components Tests" begin include("components.jl") end
    @time @safetestset "Neurograph Tests" begin include("graphs.jl") end
    @time @safetestset "Learning Tests" begin include("learning.jl") end
    @time @safetestset "DBS" begin include("dbs.jl") end
end

if GROUP == "All" || GROUP == "Advanced"
    @time @safetestset "Reinforcement Learning Tests" begin include("reinforcement_learning.jl") end
    @time @safetestset "Cort-Cort plasticity Tests" begin include("plasticity.jl") end
end


@time @safetestset "GraphDynamics vs MTK tests" begin include("GraphDynamicsTests/runtests.jl") end
