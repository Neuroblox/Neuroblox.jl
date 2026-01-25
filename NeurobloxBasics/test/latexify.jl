using NeurobloxBasics, Latexify, Symbolics, Test, ReferenceTests

@testset "Latexify regression tests" begin
    @graph g begin
        @nodes begin
            n1 = LIFNeuron()
            n2 = LIFNeuron()
        end
        @connections begin
            n1 => n2, [weight=1.0]
        end
    end
    @test_reference "plots/ltx1.txt" latexify(g)
    @test_reference "plots/ltx2.txt" latexify(JansenRit(name=:jr1))
    @test_reference "plots/ltx3.txt" latexify(JansenRit(name=:jr1, namespace=:n1₊n2₊n3))
end
