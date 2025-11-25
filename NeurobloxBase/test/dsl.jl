using NeurobloxDBS, NeurobloxPharma, NeurobloxBasics
using OrderedCollections

@testset "System creation" begin
    @named lm = LinearNeuralMass()
    @named ou = OUProcess(τ=1, σ=0.1)
    @named bold = BalloonModel()
    g = MetaDiGraph()
    add_blox!.(Ref(g), [lm, ou, bold])
    add_edge!(g, 2, 1, Dict(:weight => 1.0))
    add_edge!(g, 1, 3, Dict(:weight => 0.1))
    @named sys = system_from_graph(g)

    g2 = @graph begin
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = 1, σ = 0.1)
            bold = BalloonModel()
        end
        @connections begin
            ou => lm, [weight = 1.0]
            lm => bold, [weight = 0.1]
        end
    end
    @named sys2 = system_from_graph(g2)
    @test OrderedSet(equations(sys)) == OrderedSet(equations(sys2))
end

@testset "DSL Sanity Tests" begin
    # Allow declarations
    @test_nowarn @graph begin
        τ = 1
        σ = 0.1
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = τ, σ = σ)
            bold = BalloonModel()
        end
        @connections begin
            ou => lm, [weight = 1.0]
            lm => bold, [weight = 0.1]
        end
    end

    # Misspelled @edge
    @test_throws ErrorException @macroexpand @graph begin
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = τ, σ = σ)
            bold = BalloonModel()
        end
        @edge begin
            ou => lm, [weight = 1.0]
            lm => bold, [weight = 0.1]
        end
    end

    # Allow single edge line
    @test_nowarn @graph begin
        τ = 1
        σ = 0.1
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = τ, σ = σ)
            bold = BalloonModel()
        end
        @connections ou => lm, [weight = 1.0]
    end

    # Empty constructor
    @test_nowarn @graph
    @test_nowarn a = @graph
end
