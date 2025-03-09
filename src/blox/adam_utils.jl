using ForwardDiff
using LinearAlgebra

function compute_jacs(sol, prob, tspan; dt=1)
    p = prob.p
    function diff_function(u)
        du = similar(u)
        prob.f(du, u, p, _t)
        du
    end
    jacs = map(tspan[1]:dt:tspan[2]) do _t
        u0 = sol(_t)
        ForwardDiff.jacobian(diff_function, u0)
    end
end

function lyap(jacs::Vector{Matrix}, dt=1)
    T = size(jacs, 1)
    n = size(jacs[1], 1)
    q_store = I(n)
    lexp = zeros(n)
    lexp_counts = zeros(n)
    for t ∈ 1:T
        q, r = qr(jacs[t] * q_store)
        d = sign.(diag(r))
        d[d .== 0] .= 1
        q_store = q*Diagonal(d)

        dr = diag(Diagonal(d)*r)
        idx = dr .> 0
        lexpᵢ = log.(dr[idx])
        lexp[idx] .+= lexpᵢ
        lexp_counts[idx] .+= 1
    end

    lexp ./= lexp_counts ./ dt
end

# If we bump up ḡᵢ by a few orders of magnitude then there's increase in instability
# Need to tweak the network a bit for future experiments
ḡᵢ = 0.5
ḡₑ = 0.2

NE = 80
NI = 20

exci = [AdamPYR(name=Symbol("PYR$i"), Iₐₚₚ=rand(Normal(0.05, 0.05))) for i in 1:NE]
inhi = [AdamINP(name=Symbol("INP$i"), Iₐₚₚ=rand(Normal(0.05, 0.05))) for i in 1:NI] # bump up to 0.3

g = MetaDiGraph()

for ne ∈ exci
    for ni ∈ inhi
        if rand() > 0.9
            make_nmda_edge!(g, ne, ni)
        end
    end
end

# for ne ∈ exci
#     for ne ∈ exci
#         add_edge!(g, ne => ne; weight=ḡₑ/NE)
#     end
# end

for ne ∈ exci
    for ni ∈ inhi
        if rand() > 0.0
            add_edge!(g, ne => ni; weight=ḡₑ/NE)
        end
    end
end

for ni ∈ inhi
    for ne ∈ exci
        if rand() > 0.0
            add_edge!(g, ni => ne; weight=ḡᵢ/NI)
        end
    end
end

tspan = (0.0, 1000.0)
@named sys = system_from_graph(g, graphdynamics=true)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Tsit5(), saveat=2)

jacs = compute_jacs(sol, prob, (600, 900), dt=10)
lep = lyap(jacs, 1)
maximum(lep)