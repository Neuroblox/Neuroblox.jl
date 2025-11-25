include(joinpath(@__DIR__(), "../../../NeurobloxBase/test/test_utils.jl"))
include(joinpath(@__DIR__(), "test_suite.jl"))

const GROUP = get(ENV, "GROUP", "All")
@show GROUP

if GROUP == "All" || GROUP == "GraphDynamics1"
    basic_smoketest()
    stochastic_hh_network_tests()
end

if GROUP == "All" || GROUP == "GraphDynamics2"
    dbs_circuit_components()
    dbs_circuit()
end

