function van_der_pol(;name, θ=1.0,ϕ=0.1)
    params  = @parameters θ=θ ϕ=ϕ
    sts = @variables x(t) y(t)

    eqs = [D(x) ~ y,
           D(y) ~ θ*(1-x^2)*y - x]

    noiseeqs = [ϕ,ϕ]

    return SDESystem(eqs,noiseeqs,t,sts,params; name=name)
end
