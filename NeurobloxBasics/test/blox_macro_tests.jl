using GraphDynamics
using NeurobloxBase
using NeurobloxBasics: NeurobloxBasics as NBB, AbstractPINGNeuron
using OrdinaryDiffEqTsit5
using Test
using StochasticDiffEq
using Distributions
using Random
using LinearAlgebra
using CairoMakie

"""
Test docstring
"""
@blox struct LinearNeuralMass(; name, namespace=nothing) <: AbstractNeuralMass
    @params
    @states x = 0.0
    @inputs jcn = 0.0
    @equations begin
        D(x) = jcn
    end
end

@test string(@doc(LinearNeuralMass)) == "Test docstring\n"

"Harmonic Oscillator for test"
@blox struct HarmonicOscillator(; name, namespace=nothing,
                                ω=25*(2*pi)*0.001, ζ=1.0, k=625*(2*pi), h=35.0) <: AbstractNeuralMass
    @params ω ζ k h
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y-(2*ω*ζ*x)+ k*(2/π)*(atan((jcn)/h))
        D(y) = -(ω^2)*x
    end
end

"Jansen-Rit for test"
@blox struct JansenRit(; name,
                       namespace=nothing,
                       cortical=true,
                       τ=(cortical ? 1 : 14),
                       H=0.02,
                       λ=(cortical ? 5.0 : 400.0),
                       r=(cortical ? 0.15 : 0.1),
                       delayed=false) <: AbstractNeuralMass
    if delayed
        error("Delay systems are currently not supported")
    end
    @params τ H λ r
    @states x=1.0 y=1.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y - ((2x/τ))
        D(y) = -x/(τ^2) + (H/τ)*((2*λ)/(1 + exp(-r*(jcn))) - λ)
    end
end

"LIFNeuron for test"
@blox struct LIFNeuron(;name,
				       namespace=nothing, 
				       C=1.0,
				       Eₘ = -70.0,
				       Rₘ = 10.0,
				       τ = 10.0,
				       θ = -50.0,
				       E_syn=-70.0,
				       G_syn=0.002,
				       I_in=0.0,
                       dtmax=0.05) <: AbstractNeuron
    @params C Eₘ Rₘ τ θ E_syn G_syn I_in dtmax
    @states V=-70.0 G=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = (-(V-Eₘ)/Rₘ + I_in + jcn)/C
		D(G) = (-1/τ)*G
    end
    @discrete_events (V >= θ) => (V=Eₘ,
                                  G=G+G_syn)
end

"LIFNeuron for with continuous events for test"
@blox struct LIFNeuronCont(;name,
				           namespace=nothing, 
				           C=1.0,
				           Eₘ = -70.0,
				           Rₘ = 10.0,
				           τ = 10.0,
				           θ = -50.0,
				           E_syn=-70.0,
				           G_syn=0.002,
				           I_in=0.0) <: AbstractNeuron
    @params C Eₘ Rₘ τ θ E_syn G_syn I_in
    @states V=-70.0 G=0.0
    @inputs jcn=0.0
    @outputs G
    @equations begin
        D(V) = (-(V-Eₘ)/Rₘ + I_in + jcn)/C
		D(G) = (-1/τ)*G
    end
    @continuous_events (V - θ) => (V=Eₘ,
                                   G=G+G_syn)
end

function cont_disc_lif_test(tspan=(0.0, 9.0))
    sol1 = let
        @graph g begin
            @nodes begin
                n1 = LIFNeuron(I_in=11.0)
                n2 = LIFNeuron(I_in=13.0)
            end
            @connections begin
                n1 => n2, [weight=34.0]
                n2 => n1, [weight=12.0]
            end
        end
        prob = ODEProblem(g, [], tspan, dtmax=0.0001)
        solve(prob, Tsit5())
    end
    sol2 = let
        @graph g begin
            @nodes begin
                n1 = LIFNeuronCont(I_in=11.0)
                n2 = LIFNeuronCont(I_in=13.0)
            end
            @connections begin
                n1 => n2, [weight=34.0]
                n2 => n1, [weight=12.0]
            end
        end
        prob = ODEProblem(g, [], tspan)
        solve(prob, Tsit5())
    end
    # display(plot(sol1))
    # display(plot(sol2))
    # @info "" norm(sol1[end] - sol2[end]) / (sol1[end]⋅sol2[end]) 
    @test sol1[end] ≈ sol2[end] rtol=1e-3
end

"VanDerPol for test"
@blox struct VanDerPol(;name, namespace=nothing, θ=1.0, ϕ=0.1) <: AbstractNeuralMass
    @params θ ϕ
    @states x=0.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = θ*(1-x^2)*y - x + jcn
    end
    @computed_properties r = √(x^2 + y^2)
    @computed_properties_with_inputs begin
        jcn = jcn
    end
end

"VanDerPolNoisy for test"
@blox struct VanDerPolNoisy(;name, namespace=nothing, θ=1.0, ϕ=0.1) <: AbstractNeuralMass
    @params θ ϕ
    @states x=0.0 y=0.0
    @inputs jcn=0.0
    @outputs x
    @equations begin
        D(x) = y
        D(y) = θ*(1-x^2)*y - x + jcn
    end
    @noise_equations begin
        W(y) = ϕ
    end
    @computed_properties r = √(x^2 + y^2)
    @computed_properties_with_inputs begin
        jcn = jcn
    end
end

"PINGNeuronExci for test"
@blox struct PINGNeuronExci(;name,
                            namespace=nothing,
                            C=1.0,
                            g_Na=100.0,
                            V_Na=50.0,
                            g_K=80.0,
                            V_K=-100.0,
                            g_L=0.1,
                            V_L=-67.0,
                            I_ext=0.0,
                            τ_R=0.2,
                            τ_D=2.0) <: AbstractPINGNeuron
    @params C g_Na V_Na g_K V_K g_L V_L I_ext τ_R τ_D
    @states V=0.0 n=0.0 h=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        @setup begin
            a_m(v) = 0.32*(v+54.0)/(1.0 - exp(-(v+54.0)/4.0))
            b_m(v) = 0.28*(v+27.0)/(exp((v+27.0)/5.0) - 1.0)
            a_n(v) = 0.032*(v+52.0)/(1.0 - exp(-(v+52.0)/5.0))
            b_n(v) = 0.5*exp(-(v+57.0)/40.0)
            a_h(v) = 0.128*exp((v+50.0)/18.0)
            b_h(v) = 4.0/(1.0 + exp(-(v+27.0)/5.0))
            m∞(v)  = a_m(v)/(a_m(v) + b_m(v))
        end
        D(V) = g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn
        D(n) = (a_n(V)*(1.0 - n) - b_n(V)*n)
        D(h) = (a_h(V)*(1.0 - h) - b_h(V)*h)
        D(s) = ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
    end
end

"PINGNeuronInhib for test"
@blox struct PINGNeuronInhib(;name,
                             namespace=nothing,
                             C=1.0,
                             g_Na=35.0,
                             V_Na=55.0,
                             g_K=9.0,
                             V_K=-90.0,
                             g_L=0.1,
                             V_L=-65.0,
                             I_ext=0.0,
                             τ_R=0.5,
                             τ_D=10.0) <: AbstractPINGNeuron
    @params C g_Na V_Na g_K V_K g_L V_L I_ext τ_R τ_D
    @states V=0.0 n=0.0 h=0.0 s=0.0
    @inputs jcn=0.0
    @outputs s
    @equations begin
        @setup begin
            a_m(v) = 0.1*(v+35.0)/(1.0 - exp(-(v+35.0)/10.0))
            b_m(v) = 4*exp(-(v+60.0)/18.0)
            a_n(v) = 0.05*(v+34.0)/(1.0 - exp(-(v+34.0)/10.0))
            b_n(v) = 0.625*exp(-(v+44.0)/80.0)
            a_h(v) = 0.35*exp(-(v+58.0)/20.0)
            b_h(v) = 5.0/(1.0 + exp(-(v+28.0)/10.0))

            m∞(v) = a_m(v)/(a_m(v) + b_m(v))
        end
        D(V) = g_Na*m∞(V)^3*h*(V_Na - V) + g_K*(n^4)*(V_K - V) + g_L*(V_L - V) + I_ext + jcn
        D(n) = (a_n(V)*(1.0 - n) - b_n(V)*n)
        D(h) = (a_h(V)*(1.0 - h) - b_h(V)*h)
        D(s) = ((1+tanh(V/10.0))/2.0)*((1.0 - s)/τ_R) - s/τ_D
    end 
end

using .NBB: PINGConnection
function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronExci}, blox_dst::Subsystem{PINGNeuronInhib}, t)
    (; w, V_E) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_E - V))
end

function (c::PINGConnection)(blox_src::Subsystem{PINGNeuronInhib}, blox_dst::Subsystem{<:AbstractPINGNeuron}, t)
    (; w, V_I) = c
    (;s) = blox_src
    (;V) = blox_dst
    (; jcn = w * s * (V_I - V))
end


function test1(t_dur=20.0)
    @testset "ODE Network test" begin
        g2 = let
            local ln, ho1, ho2, jr1, jr2, lif
            @graph begin
                @nodes begin
                    ln  = NBB.LinearNeuralMass(namespace=:f₊g₊h)
                    ho1 = NBB.HarmonicOscillator()
                    ho2 = NBB.HarmonicOscillator(ω=0.0001, ζ=2.0)
                    jr1 = NBB.JansenRit(τ=1.2)
                    jr2 = NBB.JansenRit(τ=1.5)
                    lif = NBB.LIFNeuron()
                end
                @connections begin
                    ho1 => ln,  [weight=2.0]
                    ho2 => ho1, [weight=1.0]
                    jr1 => jr2, [weight=1]
                    ho1 => lif, [weight=2]
                end
            end
        end
        g1 = @graph begin
            @nodes begin
                ln  = LinearNeuralMass(namespace=:f₊g₊h)
                ho1 = HarmonicOscillator()
                ho2 = HarmonicOscillator(ω=0.0001, ζ=2.0)
                jr1 = JansenRit(τ=1.2)
                jr2 = JansenRit(τ=1.5)
                lif = LIFNeuron()
            end
            @connections begin
                ho1 => ln,  [weight=2.0]
                ho2 => ho1, [weight=1.0]
                jr1 => jr2, [weight=1]
                ho1 => lif, [weight=2]
            end
        end
        prob1 = ODEProblem(g1, [], (0.0, t_dur), [])
        sol1 = solve(prob1, Tsit5())
        
        prob2 = ODEProblem(g2, [], (0.0, t_dur), [])
        sol2 = solve(prob2, Tsit5())

        names = [
            ln.x
            ho1.x
            ho1.y
            ho2.x
            ho2.y
            jr1.x
            jr1.y
            jr2.x
            jr2.y
            lif.V
            lif.G
        ]
        @test sol1(t_dur; idxs=names) ≈ sol2(t_dur; idxs=names) rtol=1e-6
    end
end

function test2(;t_dur=20.0, seed=1234)
    local sol1
    @testset "SDE test" begin
        g1 = let
            local vdp
            @graph begin
                @nodes begin
                    vdp = NBB.VanDerPol(include_noise=true)
                end
                @connections begin
                end
            end
        end
        g2 = @graph begin
            @nodes begin
                vdp = VanDerPolNoisy()
            end
            @connections begin
            end
        end
        prob1 = SDEProblem(g1, [], (0.0, t_dur), []; seed)
        sol1 = solve(prob1, RKMil())
        
        prob2 = SDEProblem(g2, [], (0.0, t_dur), []; seed)
        sol2 = solve(prob2, RKMil())

        names = [
            vdp.x
            vdp.y
        ]
        @test sol1(t_dur; idxs=names) ≈ sol2(t_dur; idxs=names) rtol=1e-6
        @test norm(sol1(t_dur)) ≈ only(sol2(t_dur; idxs=[vdp.r]))
        @test iszero(sol2(t_dur; idxs=[vdp.jcn]))
    end
end


function ping_tests(; t_dur=2.0)
    @testset "PING Network tests" begin
        # First focus is on producing panels from Figure 1 of the PING network paper.
        # Setup parameters from the supplemental material
        μ_E = 1.5
        σ_E = 0.15
        μ_I = 0.8
        σ_I = 0.08

        # Define the PING network neuron numbers
        NE_driven = 2
        NE_other = 14
        NI_driven = 4
        N_total = NE_driven + NE_other + NI_driven

        # Extra parameters
        N=N_total
        g_II=0.2
        g_IE=0.6
        g_EI=0.6
        
        sol1 = let g=GraphSystem(), rng=Xoshiro(1234)
            # First, create the 20 driven excitatory neurons
            exci_driven = [NBB.PINGNeuronExci(name=Symbol("ED$i"), I_ext=rand(rng, Normal(μ_I, σ_I))) for i in 1:NE_driven]
            exci_other  = [NBB.PINGNeuronExci(name=Symbol("EO$i")) for i in 1:NE_other]
            inhib       = [NBB.PINGNeuronInhib(name=Symbol("ID$i"), I_ext=rand(rng, Normal(μ_I, σ_I))) for i in 1:NI_driven]

            # Create the network
            add_node!.(Ref(g), vcat(exci_driven, exci_other, inhib))

            for NE ∈ [exci_driven; exci_other]
                for NI ∈ inhib
                    add_connection!(g, NE, NI; weight = g_EI/N)
                    add_connection!(g, NI, NE; weight = g_IE/N)
                end
            end
            for NIi ∈ inhib
                for NIj ∈ inhib
                    add_connection!(g, NIi, NIj; weight=g_II/N)
                end
            end
            prob = ODEProblem(g, [], (0.0, t_dur), [])
            solve(prob, Tsit5())
        end

        sol2 = let g=GraphSystem(), rng=Xoshiro(1234)
            # First, create the 20 driven excitatory neurons
            exci_driven = [PINGNeuronExci(name=Symbol("ED$i"), I_ext=rand(rng, Normal(μ_I, σ_I))) for i in 1:NE_driven]
            exci_other  = [PINGNeuronExci(name=Symbol("EO$i")) for i in 1:NE_other]
            inhib       = [PINGNeuronInhib(name=Symbol("ID$i"), I_ext=rand(rng, Normal(μ_I, σ_I))) for i in 1:NI_driven]

            # Create the network
            add_node!.(Ref(g), vcat(exci_driven, exci_other, inhib))

            for NE ∈ [exci_driven; exci_other]
                for NI ∈ inhib
                    add_connection!(g, NE, NI; weight = g_EI/N)
                    add_connection!(g, NI, NE; weight = g_IE/N)
                end
            end
            for NIi ∈ inhib
                for NIj ∈ inhib
                    add_connection!(g, NIi, NIj; weight=g_II/N)
                end
            end
            prob = ODEProblem(g, [], (0.0, t_dur), [])
            solve(prob, Tsit5())
        end
        
        @test sol1(t_dur) ≈ sol2(t_dur) rtol=1e-10
    end
end
