using FileIO

# Loading single file - will loop to load all eventually
data = load("/Users/achesebro/Downloads/sim0001.jld2")
df = data["df"]
all_voltages = select(df, r"₊V")
time = select(df, :timestamp)

# Get all region names

all_names = names(all_voltages)
unique_names = Vector{String}()

for i in eachindex(all_names)
    name_temp = all_names[i]
    name = split(name_temp, "₊")[1]
    push!(unique_names, name)
end

unique_names = unique(unique_names)
