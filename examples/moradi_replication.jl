using Neuroblox
using OrdinaryDiffEq
using Random
using Distributions
using StatsBase
using CairoMakie


@named src = VoltageClampSource([(t=50, V=-80), (t=100, V=40)])
@named nmda = MoradiNMDAR(; τ_g=50)

g = MetaDiGraph()
add_edge!(g, src => nmda; weight=1)

@named sys = system_from_graph(g, graphdynamics=false)
tspan = (0.0, 500.0)
prob = ODEProblem(sys, [nmda.A => 4, nmda.B => 4, src.V => 40], tspan)
sol = solve(prob; saveat=0.05)

fig = Figure()
ax = Axis(fig[1,1]; ylabel="NMDA Current", xlabel="Time (ms)")
lines!(ax, sol.t, sol[nmda.I])
fig

fig = Figure()
ax = Axis(fig[1,1]; ylabel="Voltage (mV)", xlabel="Time (ms)")
lines!(ax, sol.t, sol[src.V])
fig


@named hh1 = HHNeuronExciBlox(; I_bg=0.8, G_syn=3)
@named hh2 = HHNeuronExciBlox(; I_bg=0.2, G_syn=3)
@named nmda = MoradiNMDAR(; spk_coeff=2)

g = MetaDiGraph()
add_edge!(g, hh1 => nmda; weight=1)
add_edge!(g, hh2 => nmda; weight=1, reverse=true)
add_edge!(g, nmda => hh2; weight=1)

add_edge!(g, hh1 => hh2; weight=1)

tspan = (0.0, 5000.0)
@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob; saveat=0.05)

I_ampa = sol[hh1.G] .* (0 .- sol[hh2.V])

fig = Figure()
ax = Axis(fig[1,1])

lines!(ax, sol.t, I_ampa; color=:blue)
lines!(ax,sol.t, sol[nmda.I]; color=:green)
fig


@named hh1 = HHNeuronExciBlox(; I_bg=1.8, G_syn=3)
tspan = (0.0, 5000.0)
cbs_stop = [
    [t] => [hh1.I_bg ~ 0]
    for t in 100:200:last(tspan)
]
cbs_start = [
        [t] => [hh1.I_bg ~ 2]
        for t in 200:200:last(tspan)
]

sys = system(hh1; discrete_events=vcat(cbs_start, cbs_stop))
prob = ODEProblem(sys, [], tspan)
sol = solve(prob; saveat=0.05)

lines(sol.t, sol[hh1.V])

s = detect_spikes(hh1, sol; threshold=0)
st = diff(sol.t[s.nzind])
hist(st)

using StatsBase

function bimodal_coeff(d; N=100_000)
    samples = rand(d, N)
    sk = skewness(samples)
    kr = kurtosis(samples)

    return (sk^2 + 1) / (kr + (3*(N - 1)^2) / ((N - 2) * (N - 3)))
end
