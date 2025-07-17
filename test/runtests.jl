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

if GROUP == "All" || GROUP == "Small_CS_RL_Pipeline"
    @time @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = small_corticostriatal_learning_run(N_trials=500)[100:end]
        accuracy = sum(row -> row.iscorrect, trace)/length(trace)
        @test accuracy >= 0.7
    end
end
if GROUP == "Full_CS_RL_Pipeline"
    @time @safetestset "Full Reinforcement Learning Test" begin
        include("cs_rl_testsuite.jl")
        trace = big_corticostriatal_learning_run(N_trials=500)[100:end]
        accuracy = sum(row -> row.iscorrect, trace)/length(trace)
        @test accuracy >= 0.7
    end
else
    @info "Skipping the big corticostriatal learning test. In order to run this test, launch julia with the envrionment variable `GROUP=Full_CS_RL_Pipeline`"
end
