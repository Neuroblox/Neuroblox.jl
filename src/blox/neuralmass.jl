@parameters t
D = Differential(t)

function harmonic_oscillator(;name, ω=0, ζ=0, k=0, h=0)
    params = @parameters ω=ω ζ=ζ k=k h=h
    sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
    eqs    = [D(x) ~ y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
              D(y) ~ -(ω^2)*x]
    return ODESystem(eqs, t, sts, params; name=name)
end

function jansen_rit(;name, τ=0, H=0, λ=0, r=0)
    params = @parameters τ=τ H=H λ=λ r=r
    sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
    eqs    = [D(x) ~ y - ((2/τ)*x),
              D(y) ~ -x/(τ*τ) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)]
    return ODESystem(eqs, t, sts, params; name=name)
end

function next_generation(;name, C=0, Δ=0, η_0=0, v_syn=0, alpha_inv=0, k=0)
    params = @parameters C=C Δ=Δ η_0=η_0 v_syn=v_syn alpha_inv=alpha_inv k=k
    sts    = @variables Z(t)=0.5 g(t)=1.6
    Z = ModelingToolkit.unwrap(Z)
    g = ModelingToolkit.unwrap(g)
    C, Δ, η_0, v_syn, alpha_inv, k = map(ModelingToolkit.unwrap, [C, Δ, η_0, v_syn, alpha_inv, k])
    eqs = [Equation(D(Z), (1/C)*(-im*((Z-1)^2)/2 + (((Z+1)^2)/2)*(-Δ + im*(η_0) + im*v_syn*g) - ((Z^2-1)/2)*Z))
                    D(g) ~ alpha_inv*((k/(C*pi))*(1-abs(Z)^2)/(1+Z+conj(Z)+abs(Z)^2) - g)]
    return ODESystem(eqs, t, sts, params; name=name)
end