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

orig_sim_path = "/Users/achesebro/Downloads/all_sims/orig_sims/"
resampled_sim_path = "/Users/achesebro/Downloads/all_sims/resampled_sims/"

all_files = readdir(orig_sim_path)
for i ∈ eachindex(all_files)
    orig_path = orig_sim_path * all_files[i]
    resampled_path = resampled_sim_path * all_files[i]
    orig_data = DataFrame(CSV.File(orig_path))
    collapse_data = collapse_timesteps(orig_data)
    resampled_data = resample_timeseries(collapse_data, 1)
    CSV.write(resampled_path, resampled_data)
end