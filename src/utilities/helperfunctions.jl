# define a sigmoid function
sigmoid(x::Real, r::Real) = one(x) / (one(x) + exp(-r*x))