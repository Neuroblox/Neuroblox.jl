using NeurobloxBasics
using OrdinaryDiffEqTsit5
using Test
using Random
using ForwardDiff, FiniteDiff

function test_jacobian(f, v; rtol=1e-3)
    jac_autodiff   = ForwardDiff.jacobian(f, v)
    jac_finitediff = FiniteDiff.finite_difference_jacobian(f, v)
    @test jac_autodiff ≈ jac_finitediff rtol=rtol
end

function test_derivative(f, v; rtol=1e-3)
    d_autodiff   = ForwardDiff.derivative(f, v)
    d_finitediff = FiniteDiff.finite_difference_derivative(f, v)
    @test d_autodiff ≈ d_finitediff rtol=rtol
end


@testset "Harmonic Oscillator" begin
    @named osc1 = HarmonicOscillator()
    @named osc2 = HarmonicOscillator()
    
    g = GraphSystem()
    add_connection!(g, osc1, osc2; weight=1.0)
    add_connection!(g, osc2, osc1; weight=1.0)
    sim_dur = 1e1
    prob = ODEProblem(g, [], (0.0, sim_dur),[])
    
    test_derivative(1.0) do w
        prob2 = remake(prob, p = [:w_osc1_osc2 => w])
        sol = solve(prob2, Tsit5())
        sol[osc1.x][end]
    end
    test_jacobian([1.0, 2.0, 0.25]; rtol=5e-3) do (w1, w2, ω1)
        prob2 = remake(prob, p = [:w_osc1_osc2 => w1,
                                  :w_osc2_osc1 => w2,
                                  osc1.ω => ω1])
        sol = solve(prob2, Tsit5())
        [sol[osc1.x][end], sol[osc2.x][end]]
    end
end

# TODO: More sensitivity tests

