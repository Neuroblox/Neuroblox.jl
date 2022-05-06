using Neuroblox, Test, MAT

"""
Test for ARVTarget
"""
data = matread(joinpath(@__DIR__, "lfp_test_data.mat"))
data = data["lfp"]
fs = 1000
call_rate = 150/fs
lb = 9
ub = 16
arv_estimation = Neuroblox.ARVTarget(data, lb, ub, fs, call_rate)
@test sum(arv_estimation) > 0

"""
Test for PhaseTarget
""" 
circular_loc = Neuroblox.PhaseTarget(data, lb, ub, fs)
@test all(angle.(circular_loc) .<= pi)
@test all(angle.(circular_loc) .>= -pi)

"""
Test for ControlError
"""
control_error_ARV = Neuroblox.ControlError("ARV", data, data, 9, 16, 1000, 0.150)
@test mean(control_error_ARV) .≈ 0
control_error_phase = Neuroblox.ControlError("phase", data, data, 9, 16, 1000, 0.150)
@test mean(control_error_phase) .≈ 0