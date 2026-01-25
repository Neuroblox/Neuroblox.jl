module NBLatexifyExt

using NeurobloxBase
using GraphDynamics, Latexify, Symbolics

@latexrecipe function f(sys::AbstractBlox)
    return latexify(GraphDynamics.node_equations(sys))
end

end
