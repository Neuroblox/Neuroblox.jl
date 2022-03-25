"""
test for ARVController
"""

using Neuroblox, Test, Plots, MAT

data = matread("test/lfp_test_data.mat")
data = data["lfp"]
fs = 1000
call_rate = 150/fs
lb = 9
ub = 16
filter_order = 6
controller_call_times, arv_estimation = ARVController(data, fs, call_rate, lb, ub, filter_order)

@test plot(controller_call_times, arv_estimation, linewidth=1.0)