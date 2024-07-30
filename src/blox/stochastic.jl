"""
Ornstein-Uhlenbeck process Blox

variables:
    x(t):  value
    jcn:   input 
parameters:
    τ:      relaxation time
	μ:      average value
	σ:      random noise (variance of OU process is τ*σ^2/2)
returns:
    an ODE System (but with brownian parameters)
"""
mutable struct OUBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    namespace
    stochastic
    output
    input
    odesystem
    function OUBlox(;name, namespace=nothing, μ=0.0, σ=1.0, τ=1.0)
        p = paramscoping(μ=μ, τ=τ, σ=σ)
        μ, τ, σ = p
        sts = @variables x(t)=0.0 [output=true] jcn(t)=0.0 [input=true]
        @brownian w

        eqs = [D(x) ~ -(x-μ)/τ + jcn + sqrt(2/τ)*σ*w]
        sys = System(eqs, t; name=name)
        new(namespace, true, sts[1], sts[2], sys)
    end
end

# """
# Ornstein-Uhlenbeck Coupling Blox
# This blox takes an input and multiplies that input with
# a OU process of mean μ and variance τ*σ^2/2

# This blox allows to create edges that have fluctuating weights

# variables:
#     x(t):  value
#     jcn:   input 
# parameters:
#     τ:      relaxation time
# 	μ:      average value
# 	σ:      random noise (variance of OU process is τ*σ^2/2)
# returns:
#     an ODE System (but with brownian parameters)
# """
# mutable struct OUCouplingBlox <: NeuralMassBlox
#     # all parameters are Num as to allow symbolic expressions
#     namespace
#     stochastic
#     output
#     input
#     odesystem
#     function OUCouplingBlox(;name, namespace, μ=0.0, σ=1.0, τ=1.0)
#         p = paramscoping(μ=μ, τ=τ, σ=σ)
#         μ, τ, σ = p
#         sts = @variables x(t)=0.0 [output=true] jcn(t)=0.0 [input=true]
#         @brownian w

#         eqs    = [D(x) ~ -(x-μ)/τ + sqrt(2/τ)*σ*w]
#         sys = System(eqs, t; name=name)
#         new(namespace, true, sts[2]*sts[1], sts[2], sys)
#     end
# end
