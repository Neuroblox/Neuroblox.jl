include(joinpath(@__DIR__(), "../../../NeurobloxBase/test/test_utils.jl"))
include(joinpath(@__DIR__(), "test_suite.jl"))


basic_smoketest()
basic_hh_network_tests()
ngei_test()
wta_tests()
cortical_tests()
striatum_tests()
sta_weight_namemap_test()

