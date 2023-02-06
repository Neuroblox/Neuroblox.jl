using Neuroblox, MAT, ModelingToolkit, OrdinaryDiffEq, Test

"""
Test for learningrate
    Create a vector of behavioral outcomes that are all zeros except for the last window.
    The learning rate should be zero percent for all 1:n-1 windows, and 100% for window n.
"""

outcomes = zeros(100)
outcomes[91:100] .= 1

windows = 10
learning_rate = learningrate(outcomes, windows)

@test sum(learning_rate[1:windows-1]) == 0
@test sum(learning_rate[windows]) == 100


