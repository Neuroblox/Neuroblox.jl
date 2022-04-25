"""
test for ARVController
"""

using Neuroblox, Test, MAT

data = matread(joinpath(@__DIR__, "lfp_test_data.mat"))
data = data["lfp"]
fs = 1000
call_rate = 150/fs
lb = 9
ub = 16
filter_order = 6
controller_call_times, arv_estimation = Neuroblox.ARVController(data, fs, call_rate, lb, ub, filter_order)

@test sum(arv_estimation) > 0