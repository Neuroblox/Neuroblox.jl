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
    function OUBlox(;name, μ=0.0, σ=1.0,τ=1.0)
        @parameters μ=μ τ=τ
        @variables x(t)=1.0 jcn(t)=0.0
        @brownian σ

        eqs    = [D(x) ~ -x/τ + jcn + sqrt(2/τ)*σ]
        sys = System(eqs, t; name=name)
        new(μ, σ, τ, sys.x,[sys.x],[sys.x],
            Dict(sys.x => (-1.0,1.0)),
            sys)
    end
end
