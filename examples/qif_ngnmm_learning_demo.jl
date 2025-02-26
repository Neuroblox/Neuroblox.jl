using DifferentialEquations
using Distributions
using Statistics
using Random
using Plots

function qif_ngnmm_params(;Δₑ=1.0,
                           τₘₑ=20.0,
                           Hₑ=1.3,
                           Jₑₑ=8.0,
                           Jₑᵢ=10.0,
                           Δᵢ=1.0,
                           τₘᵢ=10.0,
                           Hᵢ=-5.0,
                           Jᵢₑ=10.0,
                           Jᵢᵢ=0.0, 
                           w₁¹=1.0,
                           w₂¹=5.0,
                           w₁²=1.0,
                           w₂²=5.0,
                           curr_stim=0.0)

    return [Δₑ, τₘₑ, Hₑ, Jₑₑ, Jₑᵢ, Δᵢ, τₘᵢ, Hᵢ, Jᵢₑ, Jᵢᵢ, w₁¹, w₂¹, w₁², w₂², curr_stim]
end

p = qif_ngnmm_params()

function simple_cs_learning!(du, u, p, t)
    Δₑ, τₘₑ, Hₑ, Jₑₑ, Jₑᵢ, Δᵢ, τₘᵢ, Hᵢ, Jᵢₑ, Jᵢᵢ, w₁¹, w₂¹, w₁², w₂², curr_stim = p

    rₑ¹, Vₑ¹, rᵢ¹, Vᵢ¹, rₑ², Vₑ², rᵢ², Vᵢ², I₁, I₂ = u

    du[1] = Δₑ/(π*τₘₑ^2) + 2*rₑ¹*Vₑ¹/τₘₑ
    du[2] = (Vₑ¹^2 + Hₑ + 4*sin(5*2*π*t/1000))/τₘₑ - τₘₑ*(π*rₑ¹)^2 + Jₑₑ*rₑ¹ + Jᵢₑ*rᵢ¹ + w₁¹*I₁ + w₂¹*I₂ - 1*rₑ²
    du[3] = Δᵢ/(π*τₘᵢ^2) + 2*rᵢ¹*Vᵢ¹/τₘᵢ
    du[4] = (Vᵢ¹^2 + Hᵢ + 2*sin(5*2*π*t/1000))/τₘᵢ - τₘᵢ*(π*rᵢ¹)^2 + Jₑᵢ*rₑ¹ + Jᵢᵢ*rᵢ¹
    du[5] = Δₑ/(π*τₘₑ^2) + 2*rₑ²*Vₑ²/τₘₑ
    du[6] = (Vₑ²^2 + Hₑ + 4*sin(5*2*π*(t-100)/1000))/τₘₑ - τₘₑ*(π*rₑ²)^2 + Jₑₑ*rₑ² + Jᵢₑ*rᵢ² + w₁²*I₁ + w₂²*I₂ - 1*rₑ¹
    du[7] = Δᵢ/(π*τₘᵢ^2) + 2*rᵢ²*Vᵢ²/τₘᵢ
    du[8] = (Vᵢ²^2 + Hᵢ + 2*sin(5*2*π*(t-100)/1000))/τₘᵢ - τₘᵢ*(π*rᵢ²)^2 + Jₑᵢ*rₑ² + Jᵢᵢ*rᵢ²
    du[9] = -u[9]/30
    du[10] = -u[10]/30
end


max_time = 100000.0
stim1_times = collect(250:500:max_time)
stim2_times = collect(500:500:max_time)
all_stim_times = sort(vcat(stim1_times, stim2_times))

condtion_s1(u, t, integrator) = t ∈ stim1_times
condtion_s2(u, t, integrator) = t ∈ stim2_times

function affect_s1!(integrator)
    integrator.u[9] += 30.0
    integrator.p[15] = 1.0
end

function affect_s2!(integrator)
    integrator.u[10] += 30.0
    integrator.p[15] = 2.0
end

cb_s1 = DiscreteCallback(condtion_s1, affect_s1!)
cb_s2 = DiscreteCallback(condtion_s2, affect_s2!)

eval_times = all_stim_times[2:end]
eval_times .-= 50.0

condition_eval(u, t, integrator) = t ∈ eval_times

all_choices = []

function learn!(integrator)
    u = integrator.u
    p = integrator.p

    str1 = u[2]
    str2 = u[6]

    choice = str1 > str2

    if p[15] == 1 && choice
        p[11] += 0.1*rand()
        p[11] = min(p[11], 5.0)
        push!(all_choices, 1.0)
    elseif p[15] == 1 && !choice
        p[13] -= 0.1*rand()
        p[13] = max(p[13], -5.0)
        push!(all_choices, 0.0)
    elseif p[15] == 2 && !choice
        p[14] += 0.1*rand()
        p[14] = min(p[14], 5.0)
        push!(all_choices, 1.0)
    elseif p[15] == 2 && choice
        p[12] -= 0.1*rand()
        p[12] = max(p[12], -5.0)
        push!(all_choices, 0.0)
    end
end

cb_learn = DiscreteCallback(condition_eval, learn!)

tstop = sort(vcat(all_stim_times, eval_times))

u₀ = zeros(10)

tspan = (0.0, max_time)
cbs = CallbackSet(cb_s1, cb_s2, cb_learn)
prob = ODEProblem(simple_cs_learning!, u₀, tspan, p)
sol = solve(prob, Tsit5(), callback = cbs, tstops = tstop, saveat=1.0)
plot(all_choices)