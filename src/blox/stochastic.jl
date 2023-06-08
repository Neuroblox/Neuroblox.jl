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
    μ::Num
    σ::Num
    τ::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    system::ODESystem
    function OUBlox(;name, μ=0.0, σ=1.0, τ=1.0)
        params = @parameters μ=μ τ=τ σ=σ
        states = @variables x(t)=1.0 jcn(t)=0.0
        @brownian w

        eqs    = [D(x) ~ -(x-μ)/τ + jcn + sqrt(2/τ)*σ*w]
        sys = System(eqs, t, states, params; name=name)
        new(μ, σ, τ, sys.x,[sys.x],[sys.x],
            Dict(sys.x => (-1.0,1.0)),
            sys)
    end
end

"""
Ornstein-Uhlenbeck Coupling Blox
This blox takes an input and multiplies that input with
a OU process of mean μ and variance τ*σ^2/2

This blox allows to create edges that have fluctuating weights

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
mutable struct OUCouplingBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    μ::Num
    σ::Num
    τ::Num
    connector::Num
    noDetail::Vector{Num}
    detail::Vector{Num}
    initial::Dict{Num, Tuple{Float64, Float64}}
    system::ODESystem
    function OUCouplingBlox(;name, μ=0.0, σ=1.0, τ=1.0)
        params = @parameters μ=μ τ=τ σ=σ
        states = @variables x(t)=1.0 jcn(t)=0.0
        @brownian w
    
        eqs    = [D(x) ~ -(x-μ)/τ + sqrt(2/τ)*σ*w]
        sys = System(eqs, t, states, params; name=name)
        new(μ, σ, τ, sys.jcn*sys.x,[sys.jcn*sys.x],[sys.jcn*sys.x],
            Dict(sys.x => (-1.0,1.0)),
            sys)
    end
end
