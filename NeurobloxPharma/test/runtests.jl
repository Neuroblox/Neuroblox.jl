using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Basics"
    @safetestset "Aqua" begin 
        using Aqua, NeurobloxPharma
        Aqua.test_all(NeurobloxPharma; persistent_tasks = false)
    end
    
    @time @safetestset "Utilities" begin include("utils.jl") end
    @time @safetestset "Neurograph Tests" begin include("adjacency.jl") end
    
    @time @safetestset "GraphDynamics vs MTK tests" begin include("GraphDynamicsTests/runtests.jl") end
    @time @safetestset "Components Tests" begin include("components.jl") end
    @time @safetestset "Small Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = small_corticostriatal_learning_run(N_trials=500, seed=2025)
        accuracy = mean(trace.correct[100:end])
        @test accuracy >= 0.7
    end
end


if GROUP == "All" || GROUP == "MTK_Small_RL_Pipeline"
    @time @safetestset "Small Reinforcement Learning Test (MTK)" begin
        include("cs_rl_testsuite.jl")
        trace = small_corticostriatal_learning_run(N_trials=500, graphdynamics=false, seed=2025)
        accuracy =  mean(trace.correct[100:end])
        @test accuracy >= 0.7
    end
end

if GROUP == "All" || GROUP == "Full_CS_RL_Pipeline"
    @time @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = big_corticostriatal_learning_run(N_trials=500, seed=2025)
        accuracy =  mean(trace.correct[100:end])
        @test accuracy >= 0.7
        DA_tests(trace)
    end
end

