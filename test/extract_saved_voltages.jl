using FileIO

# Loading single file - will loop to load all eventually
data = load("/Users/achesebro/Downloads/sim0003.jld2")
df = data["df"]
all_voltages = select(df, r"₊V")
time = select(df, :timestamp)[:, 1]

# Get all region names

all_names = names(all_voltages)
unique_names = Vector{String}()

for i in eachindex(all_names)
    name_temp = all_names[i]
    name = split(name_temp, "₊")[1]
    push!(unique_names, name)
end

unique_names = unique(unique_names)

# Get unique time indices (not originally unique because of the callbacks)
time_indices = Vector{Int}()
for i in eachindex(time)
    if i == 1 || time[i] != time[i-1]
        push!(time_indices, i)
    end
end

# Extract the voltages at the unique time indices
regions = zeros(length(time_indices), length(unique_names))

for i in eachindex(unique_names)
    name = unique_names[i]
    region = select(df, Regex(name))
    temp = reduce(+, eachcol(region)) ./ ncol(region)
    regions[:, i] = temp[time_indices]
end

using Plots
plot(time[time_indices], regions, xlabel="Time", ylabel="Voltage", title="Average Voltage in Each Region")