using SafeTestsets
@safetestset "Aqua" begin 
    using Aqua, NeurobloxDBS
    Aqua.test_all(NeurobloxDBS; persistent_tasks = false)
end

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "DBS" begin include("dbs.jl") end

include(joinpath(@__DIR__(), "test_suite.jl"))

const GROUP = get(ENV, "GROUP", "All")
@show GROUP

if GROUP == "All" || GROUP == "GraphDynamics1"
    @time "GraphDynamics vs MTK tests 1" begin
        basic_smoketest()
        stochastic_hh_network_tests()
    end
end

if GROUP == "All" || GROUP == "GraphDynamics2"
    @time "GraphDynamics vs MTK tests 2" begin
        dbs_circuit_components()
        dbs_circuit()
    end
end
