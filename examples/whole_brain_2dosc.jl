# this is an example that reproduces the Virtual Brain tutorial on Resting State networks
using Neuroblox
using CSV
using DataFrames
using MetaGraphs
using DifferentialEquations
using Random
using Plots
using Statistics
using HypothesisTests

weights = CSV.read("examples/weights.csv",DataFrame)
region_names = names(weights)

wm = Array(weights)

blocks = []
for i in 1:size(wm)[1]
    push!(blocks, Neuroblox.Generic2dOscillator(name=Symbol(region_names[i]),bn=sqrt(5e-4)))
end

g = MetaDiGraph()
add_blox!.(Ref(g), blocks)
create_adjacency_edges!(g, wm)

@named sys = system_from_graph(g)
sys = structural_simplify(sys)

prob = SDEProblem(sys,rand(-2:0.1:4,76*2), (0.0, 10*60e3), [])
@time sol = solve(prob, EulerHeun(), dt=0.5, saveat=5, maxiters=5e6)
@time sol = solve(prob, SRA(), saveat=5, maxiters=5e6)
df = DataFrame(sol)
names(df)
plot(sol.t[100:end],sol[5,100:end],xlims=(1000,6000))

cs = []
for i in 1:Int((length(sol.t)-1)/1000)-1
    solv = Array(sol[1:2:end,(i-1)*1000+1:(i*1000)])'
    push!(cs,cor(solv))
end
css = stack(cs)
cssath = atanh.(css)
for i in 1:size(cssath)[1]
    cssath[i,i,:] = zeros(size(cssath)[3])
end
p = zeros(76,76)
for i in 1:76
    for j in 1:76
        p[i,j] = pvalue(OneSampleTTest(css[i,j,:]))
    end
end
p

plot(heatmap(wm,aspect_ratio = :equal), heatmap(log10.(p) .* (p .< 0.05),aspect_ratio = :equal))

heatmap(log10.(p) .* (p .< 0.05),aspect_ratio = :equal)
heatmap(wm)
