using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Basics"
    @time @safetestset "Small Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = small_corticostriatal_learning_run(N_trials=500, seed=2025)
        accuracy = mean(trace.correct[100:end])
        @test accuracy >= 0.7
    end
end

if GROUP == "All" || GROUP == "Full_CS_RL_Pipeline" 
    @time @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        using UnicodePlots
        trace = big_corticostriatal_learning_run(N_trials=500, seed=2025)
        accuracy = mean(trace.correct[100:end])
        @test accuracy >= 0.7
        DA_tests(trace)
    end
else
    @info "Skipping the big corticostriatal learning test. In order to run this test, launch julia with the envrionment variable `GROUP=Full_CS_RL_Pipeline`"
end
