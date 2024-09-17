using CSV, DataFrames
using Interpolations

function collapse_timesteps(data)
    new_data = DataFrame()
    for i ∈ eachindex(data.t)
        if i > 1 && data.t[i] != data.t[i-1]
            push!(new_data, data[i, :])
        end
    end
    return new_data
end

function resample_timeseries(data, dt=1)
    new_data = DataFrame()
    new_t = 0:dt:data.t[end]
    new_data.t = new_t
    for i ∈ eachindex(Array(data[1, 2:end]))
        temp_interp = linear_interpolation(data.t, data[:, i+1], extrapolation_bc=Line())
        new_data[!, names(data)[i+1]] = temp_interp(new_t)
    end
    return new_data
end 


data = DataFrame(CSV.File("/Users/achesebro/Downloads/all_sims/sol_trial_1.csv"))
