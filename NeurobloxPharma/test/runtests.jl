using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Basics"
    @safetestset "Aqua" begin 
        using Aqua, NeurobloxPharma
        Aqua.test_all(NeurobloxPharma; persistent_tasks = false)
    end
    @safetestset "Utilities" begin include("utils.jl") end
    @safetestset "Latexify" begin
        include("latexify.jl")
    end
    @safetestset "Neurograph Tests" begin include("adjacency.jl") end
    @safetestset "Components Tests" begin include("components.jl") end
    @safetestset "Small Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = small_corticostriatal_learning_run(N_trials=400, seed=2025, save_everystep=false)
        accuracy = mean(trace.correct[100:end])
        @test accuracy >= 0.7
    end
end

if GROUP == "All" || GROUP == "Receptors"
    @safetestset "Receptors Tests" begin
        include("receptors_test.jl")
    end
end

if GROUP == "All" || GROUP == "Full_CS_DA_Tests"
    @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = big_corticostriatal_learning_run(N_trials=100, seed=2025, save_everystep=false)
        DA_tests(trace)
    end
end

if GROUP == "All" || GROUP == "Full_CS_RL_Pipeline"
    @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = big_corticostriatal_learning_run(N_trials=400, seed=2025, save_everystep=false,
                                                 scheduler=PolyesterScheduler())
        accuracy =  mean(trace.correct[100:end])
        @test accuracy >= 0.7
        DA_tests(trace)
    end
end
