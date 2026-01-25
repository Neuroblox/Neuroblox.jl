using NeurobloxPharma, Latexify, Symbolics, Test, ReferenceTests

@testset "Latexify regression tests" begin
    @graph g begin
        @nodes begin
            n1 = HHNeuronExci()
            n2 = HHNeuronExci()
        end
        @connections begin
            n1 => n2, [weight=1.0]
        end
    end

    #@test_reference "plots/connection_equations_intermediates_latexify.txt" latexify(GraphDynamics.connection_equations(g, n1, n2))
end
