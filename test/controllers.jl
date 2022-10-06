using Neuroblox, Test, MAT, Statistics

"""
Test for ARVTarget
"""
data = matread(joinpath(@__DIR__, "lfp_test_data.mat"))
data = data["lfp"]
fs = 1000
call_rate = 150/fs
lb = 9
ub = 16
order = 4
control_error_ARV  = Neuroblox.ControlError("ARV", data, data, lb, ub, fs, order, call_rate)

tol = 0.25
@test Statistics.mean(control_error_ARV) < tol
@test Statistics.mean(control_error_ARV) > -tol

"""
Test for CDVTarget
""" 
control_error_phase = Neuroblox.ControlError("CDV", data, data, lb, ub, fs, order, call_rate)
@test Statistics.mean(control_error_phase) < tol
@test Statistics.mean(control_error_phase) > -tol