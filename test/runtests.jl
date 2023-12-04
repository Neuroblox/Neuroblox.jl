using SafeTestsets

@time @safetestset "Components Tests" begin include("components.jl") end
@time @safetestset "Neurograph Tests" begin include("graphs.jl") end

# Commenting out because we no longer use ODEfromGraph for neural masses
#@time @safetestset "ODE from Graph Tests" begin include("ode_from_graph.jl") end

# @time @safetestset "Neural Signal Measurement Models Tests" begin include("measurementmodels.jl") end
# @time @safetestset "Spectral Utilities Tests" begin include("spectraltools.jl") end
@time @safetestset "Data Fitting Tests" begin include("datafitting.jl") end 
@time @safetestset "ODE from Graph and simulate" begin include("graph_to_dataframe.jl") end
@time @safetestset "Learning Tests" begin include("learning.jl") end
@time @safetestset "Control Tests" begin include("controllers.jl") end
@time @safetestset "Source Tests" begin include("source_components.jl") end
@time @safetestset "Reinforcement Learning Tests" begin include("reinforcement_learning.jl") end
@time @safetestset "Cort-Cort plasticity Tests" begin include("plasticity.jl") end
# fitting tests should be at the end since they take the longest
# removing them for now until we have real fitting tests
# @time @safetestset "Fitting Tests" begin include("fitting.jl") end
# @time @safetestset "Bayesian Fitting Tests" begin include("bayesian_fitting.jl") end

# Extra tests for comparison across blox versions
#@time @safetestset "New Jansen-Rit Tests" begin include("jansen_rit_component_tests_new_timing.jl") end
#@time @safetestset "Old Jansen-Rit Tests" begin include("old_component_tests/jansen_rit_tests.jl") end
#@time @safetestset "Old Wilson-Cowan Tests" begin include("old_component_tests/wilson_cowan_tests.jl") end 
#@time @safetestset "New Larter-Breakspear Tests" begin include("new_LB_blox.jl") end
#@time @safetestset "Old Larter-Breakspear Tests" begin include("old_component_tests/larter_breakspear_tests.jl") end
