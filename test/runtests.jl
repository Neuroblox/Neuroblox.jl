using SafeTestsets

@safetestset "Aqua" begin 
    using Aqua, Neuroblox
    Aqua.test_all(Neuroblox;
                  persistent_tasks = false)
end

@time @safetestset "Integration tests" begin include("integration.jl") end
