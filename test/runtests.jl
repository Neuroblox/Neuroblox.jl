using SafeTestsets

@safetestset "Aqua" begin 
    using Aqua, Neuroblox
    Aqua.test_all(Neuroblox;
                  persistent_tasks = false,
                  stale_deps=(; ignore=[:NeurobloxSynapticBlox]))
end

@time @safetestset "Integration tests" begin include("integration.jl") end
