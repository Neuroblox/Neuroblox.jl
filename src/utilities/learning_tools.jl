"""
This function computes learning rate.
It has the following inputs:
    outcomes: vector of 1's and 0's for behavioral outcomes
    windows: number of windows to split the outcome data into
And the following outputs:
    rate: the learning rate across each window
"""
function learningrate(outcomes, windows)
    bins = Int(floor(length(outcomes)*(1/windows)))
    lrate = zeros(windows)
    for i in 1:windows
        if i == 1
            lrate[i] = sum(outcomes[1:bins*i])
        else
            lrate[i] = sum(outcomes[bins*(i-1):bins*i])
        end
    end
    learning_rate = 100 .* (lrate ./ bins)
    return learning_rate
end