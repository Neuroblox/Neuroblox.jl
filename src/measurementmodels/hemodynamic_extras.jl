using Random, SpecialFunctions

@parameters t
D = Differential(t)

"""
Woolrich, Behrens and Smith. 2003.

Specify widths (h₁, h₂, h₃, h₄) in seconds. 
Amplitudes (f₁, f₂) relative to HRF height.
Output is HRF kernel in ms.

"""

function HRFFourHalfCosine(; h₁=nothing, h₂=nothing, h₃=nothing, h₄=nothing, f₁=nothing, f₂=nothing)

    h₁ = isnothing(h₁) ? 2*rand() : h₁
    h₂ = isnothing(h₂) ? 4*rand()+2 : h₂
    h₃ = isnothing(h₃) ? 4*rand()+2 : h₃
    h₄ = isnothing(h₄) ? 6*rand()+2 : h₄
    f₁ = isnothing(f₁) ? 0 : f₁
    f₂ = isnothing(f₂) ? 0.5*rand() : f₂

    t₁ = 0:0.001:h₁
    t₂ = 0:0.001:h₂
    t₃ = 0:0.001:h₃
    t₄ = 0:0.001:h₄

    cos₁ = 0.5.*f₁.*(cos.(π.*(t₁./h₁)) .- 1)
    cos₂ = ((0.5*(f₁+1)).*(-cos.(π.*(t₂./h₂)) .+ 1)) .- f₁
    cos₃ = ((0.5*(f₂+1)).*(cos.(π.*(t₃./h₃)))) .+ ((0.5*(1-f₂)))
    cos₄ = (0.5.*f₂.*(-cos.(π.*(t₄./h₄)) .+ 1)) .- f₂

    hrf = vcat(cos₁, cos₂, cos₃, cos₄)

end

"""
Lindquist et al. 2012.

All parameters are in seconds.
Output is in ms.
If you specify a change in any of the parameters, also change the tlen to ensure the kernel is wide enough.
It will throw an error if you don't.

"""

function HRFDoubleGamma(; A=nothing, α₁=nothing, α₂=nothing, β₁=nothing, β₂=nothing, c=nothing, tlen=nothing)

    if (!isnothing(A) || !isnothing(α₁) || !isnothing(α₂) || !isnothing(β₁) || !isnothing(β₂) || !isnothing(c)) && isnothing(tlen)
        throw(DomainError(nothing, "If you specify a different parameter, you need to specify tlen as well to ensure the kernel is wide enough."))
    end

    A = isnothing(A) ? 1 : A
    α₁ = isnothing(α₁) ? 6 : α₁
    α₂ = isnothing(α₂) ? 16 : α₂
    β₁ = isnothing(β₁) ? 1 : β₁
    β₂ = isnothing(β₂) ? 1 : β₂
    c = isnothing(c) ? 1.0/6.0 : c
    tlen = isnothing(tlen) ? 30 : tlen

    t = 0:0.001:tlen
    
    hrf = A.*((((t.^(α₁-1)).*(β₁^α₁).*exp.(-β₁.*t))./gamma(α₁)) .- (c.*((t.^(α₂-1)).*(β₂^α₂).*exp.(-β₂.*t)./ gamma(α₂))))

end


"""
### Input variables ###
adaptation of the Hemodynamics blox in fmri.jl
"""
struct BalloonModel <: ObserverBlox
    params
    output
    jcn
    odesystem
    namespace
    function BalloonModel(;name, lnκ=0.0, lnτ=0.0)
        #= hemodynamic parameters
            H(1) - signal decay                                   d(ds/dt)/ds)
            H(2) - autoregulation                                 d(ds/dt)/df)
            H(3) - transit time                                   (t0)
            H(4) - exponent for Fout(v)                           (alpha)
            H(5) - resting state oxygen extraction                (E0)
        =#

        H = [0.64, 0.32, 2.00, 0.32, 0.4]
        p = progress_scope(@parameters lnκ=lnκ lnτ=lnτ)  # progress scope if needed
        #p = compileparameterlist(lnκ=p[1], lnτ=p[2])  # finally compile all parameters
        lnκ, lnτ = p  # assign the modified parameters
        
        sts = @variables s(t)=1.0 lnf(t)=1.0 lnν(t)=1.0 [output=true, description="hemodynamic_observer"] lnq(t)=1.0 [output=true, description="hemodynamic_observer"] jcn(t)=0.0 [input=true]

        eqs = [
            D(s)   ~ jcn/1e2 - H[1]*exp(lnκ)*s - H[2]*(exp(lnf) - 1),
            D(lnf) ~ s / exp(lnf),
            D(lnν) ~ (exp(lnf) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν)),
            D(lnq) ~ (exp(lnf)/exp(lnq)*((1 - (1 - H[5])^(exp(lnf)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
        ]
        sys = System(eqs, name=name)
        new(p, Num(0), sts[5], sys, nothing)
    end
end


# This doesn't work for some good reasons. You can't call system from graph and expect to call it again later
struct CompoundHemo <: CompoundNOBlox
    params
    output
    jcn
    odesystem
    namespace
    function CompoundHemo(massChoice; name, lnκ=0.0, lnτ=0.0) #ONLY WORKS WITH DEFAULT PARAMETERS FOR NOW
        p = progress_scope(@parameters lnκ lnτ)
        @named hemo = BalloonModel(;lnκ=p[1], lnτ=p[2])
        @named nmm = massChoice()
        @variables jcn(t)
        g = MetaDiGraph()
        add_blox!(g, nmm)
        add_blox!(g, hemo)
        add_edge!(g, 1, 2, Dict(:weight => 1.0))
        linhemo = system_from_graph(g; name=name)
        new(p, states(linhemo)[1], states(linhemo)[2], linhemo, nothing)
    end
end

struct LinHemoCombo <: CompoundNOBlox
    params
    output
    jcn
    odesystem
    namespace
    function LinHemoCombo(;name, lnκ=0.0, lnτ=0.0)
        p = progress_scope(@parameters lnκ=lnκ lnτ=lnτ) 
        lnκ, lnτ = p  # assign the modified parameters
        H = [0.64, 0.32, 2.00, 0.32, 0.4]

        sts = @variables nmm₊x(t)=0.0 [output=true] nmm₊jcn(t)=0.0 [input=true] hemo₊s(t)=1.0 hemo₊lnf(t)=1.0 hemo₊lnν(t)=1.0 hemo₊lnq(t)=1.0 hemo₊jcn(t)=0.0
        eqs = [D(nmm₊x) ~ nmm₊jcn,
               D(hemo₊s)   ~ nmm₊x - H[1]*exp(lnκ)*hemo₊s - H[2]*(exp(hemo₊lnf) - 1),
               D(hemo₊lnf) ~ hemo₊s / exp(hemo₊lnf),
               D(hemo₊lnν) ~ (exp(hemo₊lnf) - exp(hemo₊lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(hemo₊lnν)),
               D(hemo₊lnq) ~ (exp(hemo₊lnf)/exp(hemo₊lnq)*((1 - (1 - H[5])^(exp(hemo₊lnf)^-1))/H[5]) - exp(hemo₊lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ))
              ]
        sys = System(eqs, name=name)
        
        new(p, sts[1], sts[2], sys, nothing)
    end
end


struct AlternativeBalloonModel <: ObserverBlox
    params
    output
    jcn
    odesystem
    namespace
    function AlternativeBalloonModel(;name, κ=0.65, α=0.32, τ=0.98, ρ=0.34, V₀=0.02, γ=0.41)
        p = progress_scope(@parameters κ=κ α=α τ=τ ρ=ρ V₀=V₀ γ=γ)  # progress scope if needed
        κ, α, τ, ρ, V₀, γ = p  # assign the modified parameters
        sts = @variables s(t)=1.0 f(t)=1.0 v(t)=1.0 q(t)=1.0 b(t)=0.0 [output=true] jcn(t)=0.0 [input=true]
        eqs = [
            D(s) ~ jcn/1e2 - κ*s - γ*(f - 1),
            D(f) ~ s,
            D(v) ~ (f - v^(1/α))/τ,
            D(q) ~ (((f*(1 - (1 - ρ^(1/f))/ρ)) - (v^(1/α)*q)/v))/τ,
            b ~ V₀ * ((7*ρ*(1 - q)) + (2*(1-q/v)) + ((2*ρ - 0.2)*(1 - v)))
        ]
        sys = System(eqs, name=name)
        new(p, Num(0), sts[5], sys, nothing)
    end
end

mutable struct JRHemo <: AbstractComponent
    connector::Num
    bloxinput::Num
    odesystem::ODESystem
    function JRHemo(;name, lnκ=0.0, lnτ=0.0)
        params = progress_scope(lnκ, lnτ)
        @named hemo = Hemodynamics(;lnκ=params[1], lnτ=params[2])
        @named nmm = JansenRitCBlox()
        @variables jcn(t)

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => nmm, :jcn => jcn))
        add_vertex!(g, :blox, hemo)
        add_edge!(g, 1, 2, :weight, 1.0)
        linhemo = ODEfromGraph(g; name=name)
        new(linhemo.nmm₊x, linhemo.jcn, linhemo)
    end
end