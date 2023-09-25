using Random, SpecialFunctions, Plots


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
