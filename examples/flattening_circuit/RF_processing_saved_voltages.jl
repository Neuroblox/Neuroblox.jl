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

concat_file_path = "/Users/achesebro/Downloads/all_sims/concatenated_sims.csv"
all_files = readdir(resampled_sim_path)
first_data = DataFrame(CSV.File(resampled_sim_path * all_files[1]))
all_data = first_data

for i ∈ 2:lastindex(all_files)
    next_data = DataFrame(CSV.File(resampled_sim_path * all_files[i]))
    all_data = vcat(all_data, next_data)
end

new_t = 0:lastindex(all_data.t)-1
all_data.t .= collect(new_t)

CSV.write(concat_file_path, all_data)

all_data = CSV.read(concat_file_path, DataFrame)

using DSP, Distributions, StatsBase

function HRF(x,rtu=6.0,δ1=6.0,δ2=16.0,τ1=1.0,τ2=1.0,C=0.0)
    return pdf.(Gamma(δ1,τ1),x)-pdf.(Gamma(δ2,τ2),x)/rtu + C
end


t_hrf = 0:0.001:32
hrf = HRF.(t_hrf)

hrf_data = DataFrame()

for i ∈ names(all_data)[2:end]
    next_data = all_data[:, i]
    next_data = StatsBase.transform(fit(UnitRangeTransform, next_data), next_data)
    conv_data = conv(next_data, hrf)
    hrf_data[!, i] = conv_data
end

hrf_path = "/Users/achesebro/Downloads/all_sims/hrf_sims.csv"
CSV.write(hrf_path, hrf_data)
