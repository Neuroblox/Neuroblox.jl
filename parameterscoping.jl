using ModelingToolkit

@variables t
D = Differential(t)

@parameters a b
p = [a              # a is a local variable
    ParentScope(b)]  # b is a variable that belongs to one level up in the hierarchy
    b = p[2]
sts = @variables x(t), y(t)

level0 = ODESystem(Equation[D(x) ~ a*x + b*y], t, sts, p; name = :level0)
level1 = ODESystem(Equation[D(y) ~ a*x + b*y, x~level0.x, y~level0.y], t, sts, p; name = :level1) ∘ level0
level1 = structural_simplify(level1)

parameters.([level0, level1])
equations.([level0, level1])


#level0₊a
#b
#c
#level0₊d
#level0₊e
#f
level2 = ODESystem(Equation[D(z) ~ a + b + c + d + e + f], t, [], []; name = :level2) ∘ level1
parameters(level2)
#level1₊level0₊a
#level1₊b
#c
#level0₊d
#level1₊level0₊e
#f
level3 = ODESystem(Equation[D(jcn) ~ a + b + c + d + e + f], t, [], []; name = :level3) ∘ level2
parameters(level3)


function parameter_list(para_dict)
    params = []
    for (n,v) in para_dict
         if typeof(v) == Num
            push!(params, v)
        else
            push!(params, (@parameters $n = v)[1])
        end
    end
    return params
end

function van_der_pol(;name, θ=1.0)
    par = parameter_list(Dict(:θ => θ))
    sts = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-x^2)*y - x]

    return ODESystem(eqs,t,sts,par; name=name)
end

function van_der_pol_coupled(;name, theta=1.0)
    @named VP1 = van_der_pol(θ=theta)
    @named VP2 = van_der_pol(θ=theta)
    @variables jcn(t)

    eqs = [VP1.jcn ~ VP2.x,
           VP2.jcn ~ jcn]
    sys = [VP1,VP2]

    return compose(ODESystem(eqs;name=:connected),sys; name=name)
end

@parameters θ=1.0
@named VP = van_der_pol_coupled(theta=θ)

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
J = @RuntimeGeneratedFunction(generate_jacobian(VP)[1])
J(states(VP), parameters(VP), t)



# pars = foo[][3].name
sts = [v for v in values(foo[][2])]
fun = generate_jacobian(neuronmodel, expression = Val{true})[1]
tmpfun = Dict(:a => p -> fun(sts, p, t))
fun(foo[][3].mean)

bar[][1](foo[][3].mean)

foo[][]
bar[][1]
### Test 
bar[][1]((p->p.value).(bar[][2]))

foo[][3].mean

bar[][2]
bar[][1]()




function oneode_sys(;name, alpha=20.0, beta=-0.01)
    par = parameter_list(Dict(:α => alpha, :β => beta))
    @show par
    sts = @variables x(t) jcn(t)

    eqs = [D(x) ~ α - x^2 + β*jcn]

    return ODESystem(eqs, t, sts, par; name=name)
end

function twoode_sys(;name, theta=1.0)
    par = parameter_list(Dict(:θ => theta))
    sts = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0

    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*y - x^2]

    return ODESystem(eqs, t, sts, par; name=name)
end


function coupled_sys(;name, alpha=20.0, theta=1.0)
    @named S1 = oneode_sys(alpha=alpha)
    @named S2 = twoode_sys(theta=theta)
    @variables jcn(t)

    eqs = [S1.jcn ~ S2.x,
           S2.jcn ~ jcn]
    sys = [S1, S2]

    return compose(ODESystem(eqs; name=:connected), sys; name=name)
end

@parameters θ=1.0
@named VP = coupled_sys(theta=2.0, alpha=20.0)