using Neuroblox, DifferentialEquations, DataFrames, Test, Distributions, Statistics, LinearAlgebra, Graphs, MetaGraphs, Random, CSV

# @variables t
# D = Differential(t)

# continuous_events = [[V₁~Vₜ] => [V₁~Vᵣ, s_cum1~s_cum1+sⱼ]
#                      [V₂~Vₜ] => [V₂~Vᵣ, s_cum2~s_cum2+sⱼ]]

# sts = @variables V₁(t)=-2.0 V₂(t)=-1.9 I₁(t)=0.0 I₂(t)=0.0 s₁(t)=0.0 s₂(t)=0.0 s_cum1(t)=0.0 s_cum2(t)=0.0
# params = @parameters J=15 η=-5 τₛ=1 Vₜ=2 Vᵣ=-2 sⱼ=1.0

# eqs = [
#     D(V₁) ~ V₁^2 + I₁,
#     D(I₁) ~ η + J*s₁,
#     D(s₁) ~ -s₁/τₛ + (s_cum1+s_cum2)/2,
#     D(s_cum1) ~ -s_cum1/τₛ,
#     D(V₂) ~ V₂^2 + I₂,
#     D(I₂) ~ η + J*s₂,
#     D(s₂) ~ -s₂/τₛ + (s_cum1+s_cum2)/2,
#     D(s_cum2) ~ -s_cum2/τₛ
# ]

# @named odesys = ODESystem(eqs, t, sts, params; continuous_events) 
# hmm2 = structural_simplify(odesys)
# prob = ODEProblem(odesys, Pair[], (0.0, 100.0))
# sol = solve(prob, Tsit5(), saveat=0.1)

sts = @variables V1(t)=1 I1(t)=0 s1(t)=0 scum1(t)=2 V2(t)=1 I2(t)=0 s2(t)=0 scum2(t)=2
@variables t
D = Differential(t)

continuous_events = [[V1 ~ Vₜ] => [V1 ~ Vᵣ, scum1 ~ scum1+sⱼ]
                     [V2 ~ Vₜ] => [V2 ~ Vᵣ, scum2 ~ scum2+sⱼ]]
params = @parameters J=1 η=-5 τₛ=1 Vₜ=2 Vᵣ=-2 sⱼ=1.0

eqs = [
    D(V1) ~ V1^2 + I1,
    D(I1) ~ η + J*s1,
    D(s1) ~ -s1/τₛ + (scum1+scum2)/2,
    D(scum1) ~ -scum1/τₛ,
    D(V2) ~ V2^2 + I2,
    D(I2) ~ η + J*s2,
    D(s2) ~ -s2/τₛ + (scum1+scum2)/2,
    D(scum2) ~ -scum2/τₛ
]
@named ball = ODESystem(eqs, t, sts, params; continuous_events)

ball = structural_simplify(ball)

tspan = (0.0, 10.0)
prob = ODEProblem(ball, Pair[], tspan)

sol = solve(prob, Tsit5())

@variables t
D = Differential(t)

mutable struct IzhNeuronBlox
    connector::Num
    odesystem::ODESystem
    function IzhNeuronBlox(;name, α=0.6215, η=0.12, Iₑ=0.0, a=0.0077, b=-0.0062, vₚ=200, vᵣ=-200, wⱼ=0.0189, sⱼ=1.2308, eᵣ=1.0, gₛ=1.2308, τₛ=2.6)
        sts = @variables V(t)=0.0 w(t)=0.0 s(t)=0.0 s_cumulative(t)=0.0 jcn(t)=0.0
        ps = @parameters α=α η=η Iₑ=Iₑ a=a b=b vₚ=vₚ vᵣ=vᵣ wⱼ=wⱼ sⱼ=sⱼ eᵣ=eᵣ gₛ=gₛ τₛ=τₛ
        eqs = [
            D(V) ~ V*(V-α)-w+η+Iₑ+gₛ*s*(eᵣ-V),
            D(w) ~ a*(b*V-w),
            D(s) ~ -s/τₛ + jcn,
            D(s_cumulative) ~ -s_cumulative/τₛ
        ]
        ev = [V~vₚ] => [V~vᵣ, w~w+wⱼ, s_cumulative~s_cumulative+sⱼ]
        odesys = ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
        new(odesys.s_cumulative, odesys)
    end
end

mutable struct IzhNeuronBlox
    connector::Num
    odesystem::ODESystem
    function IzhNeuronBlox(;name, α=0.6215, η=0.12, Iₑ=0.0, a=0.0077, b=-0.0062, vₚ=200, vᵣ=-200, wⱼ=0.0189, sⱼ=1.2308, eᵣ=1.0, gₛ=1.2308, τₛ=2.6)
        sts = @variables V(t)=0.0 w(t)=0.0 s(t)=0.0 jcn(t)=0.0
        ps = @parameters α=α η=η Iₑ=Iₑ a=a b=b vₚ=vₚ vᵣ=vᵣ wⱼ=wⱼ sⱼ=sⱼ eᵣ=eᵣ gₛ=gₛ τₛ=τₛ
        eqs = [
            D(V) ~ V*(V-α)-w+η+Iₑ+gₛ*s*(eᵣ-V),
            D(w) ~ a*(b*V-w),
            D(s) ~ -s/τₛ + jcn*sⱼ,
        ]
        ev = [V~vₚ] => [V~vᵣ, w~w+wⱼ]
        odesys = ODESystem(eqs,t,sts,ps,continuous_events=[ev];name=name)
        new(1/(1+exp(-(odesys.V - 100))), odesys)
    end
end

N = 100

blox = []
for i = 1:N
    next_neuron = IzhNeuronBlox(;name=Symbol("neuron$i"), η=rand(Cauchy(0.12, 0.02)))
    push!(blox, next_neuron)
end

adj_matr = ones(N, N)
#adj_matr[adj_matr .< 0.001] .= 0
#adj_matr[adj_matr .> 0] .= 1

all_sys = []
for i = 1:N
    push!(all_sys, blox[i].odesystem)
end

all_connects = []
for i = 1:N
    push!(all_connects, blox[i].connector)
end

connection_eqs = []
for i = 1:N
    push!(connection_eqs, blox[i].odesystem.jcn ~ sum(adj_matr[:, i] .* all_connects/N))
end

@named connection_system = ODESystem(connection_eqs, t)
@named final_sys = compose(connection_system, all_sys...)
final_sys = structural_simplify(final_sys)
prob = ODEProblem(final_sys, [], (0.0, 500.0))
sol = solve(prob, Tsit5(), saveat=1)
df = DataFrame(sol)
CSV.write("/Users/achesebro/Downloads/test_100N_fixed_difftau.csv", df)