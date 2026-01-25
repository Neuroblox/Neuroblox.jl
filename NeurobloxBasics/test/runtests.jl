using SafeTestsets
using Test

@safetestset "Aqua" begin 
    using Aqua, NeurobloxBasics
    Aqua.test_all(NeurobloxBasics; persistent_tasks = false)
end

@time @safetestset "Utilities" begin include("utils.jl") end
@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Sensitivity Tests" begin include("sensitivity_tests.jl") end
@time @safetestset "@blox tests" begin
    include("blox_macro_tests.jl")
    @testset "@blox macro tests" begin
        test1()
        test2()
        ping_tests()
        cont_disc_lif_test()
    end
end
