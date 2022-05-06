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
circular_loc = Neuroblox.PhaseTarget(data)
@test angle(circular_loc) <= pi
@test angle(circular_loc) >= pi

"""
Test for ControlError
"""
function ControlError(type, target, actual, lb, ub, fs, call_rate)

    control_bin = call_rate*fs
    if type == ARV
        arv_target = Neuroblox.ARVTarget(target, lb, ub, fs, control_bin)
        arv_actual = Neuroblox.ARVTarget(actual, lb, ub, fs, control_bin)
        control_error = arv_target - arv_actual
    end

    if type == phase
        phi_target = Neuroblox.PhaseTarget(target)
        phi_actual = Neuroblox.PhaseTarget(actual)
        control_error = angle.(phi_target./exp.(phi_actual))
    end

    return control_error
end
control_error_ARV = Neuroblox.ControlError("ARV", data, data, 9, 16, 1000, 0.150)
@test control_error_ARV ≈ 0
control_error_phase = Neuroblox.ControlError("phase", data, data, 9, 16, 1000, 0.150)
@test control_error_phase ≈ 0