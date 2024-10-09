include(joinpath(@__DIR__(), "test_suite.jl"))

const GROUP = get(ENV, "GROUP", "All")
@show GROUP

if GROUP == "All" || GROUP == "GraphDynamics1"
    basic_smoketest()
    neuron_and_neural_mass_comparison_tests()
    basic_hh_network_tests()
    stochastic_hh_network_tests()
    ngei_test()
    kuramato_test()
end

if GROUP == "All" || GROUP == "GraphDynamics2"
    wta_tests()
    dbs_circuit_components()
    dbs_circuit()
    discrete()
end
if GROUP == "All" || GROUP == "GraphDynamics3"
    striatum_tests()
    lif_exci_inh_tests()
    decision_making_test()
    ping_tests()
end
