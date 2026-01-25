using NeurobloxBase: NeurobloxBase, BasicConnection, EventConnection, namespaceof
using NeurobloxDBS, NeurobloxPharma, NeurobloxBasics
using OrderedCollections
using GraphDynamics: GraphDynamics, Subsystem, SubsystemParams, SubsystemStates, GraphSystem, to_subsystem, nodes
using OrdinaryDiffEqTsit5
using Test

struct MyEventConnection
    weight::Float64
    event_times::NamedTuple
end

struct MyConnection
    weight::Float64
end

function compare_graph_solutions(g1::GraphSystem, g2::GraphSystem, tspan, op = [])
    prob1 = ODEProblem(g1, op, tspan)
    prob2 = ODEProblem(g2, op, tspan)
    sol1 = solve(prob1, Tsit5());
    sol2 = solve(prob2, Tsit5());
    @test sol1 == sol2
end

@testset "@graph: Sanity Tests" begin
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
    @test_throws UndefVarError @macroexpand @graph begin
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
    @test_nowarn g = @graph begin
        τ = 1
        σ = 0.1
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = τ, σ = σ)
            bold = BalloonModel()
        end
        @connections ou => lm, [weight = 1.0]
    end
    g = @graph begin
        τ = 1
        σ = 0.1
        @nodes begin
            lm = LinearNeuralMass()
            ou = OUProcess(τ = τ, σ = σ)
            bold = BalloonModel()
        end
        @connections ou => lm, [weight = 1.0]
    end
    @test length(collect(GraphDynamics.connections(g))) == 1

    # Empty constructor
    @test_nowarn @graph
    @test_nowarn a = @graph

    # Named nodes
    g2 = @graph begin
        @nodes begin
            neuron = HHNeuronInhib(name = :weird_name)
        end
    end
    @test namespaced_nameof(neuron) == :weird_name

    # Namespace and naming
    @graph neuralnet begin
        @nodes begin
            ping_exci1 = PINGNeuronExci()
            ping_exci2 = PINGNeuronExci(; namespace = :weirdnet)
            ping_inhib = PINGNeuronInhib(; name = :weirdneuron)
            hhneuron = HHNeuronExci(; namespace = :weirdnet, name = :weirdneuron)
        end
    end
    @test neuralnet isa GraphSystem
    @test nameof(ping_exci1) == :ping_exci1
    @test namespaceof(ping_exci2) == :weirdnet
    @test nameof(ping_inhib) == :weirdneuron
    @test namespaceof(hhneuron) == :weirdnet
end

@testset "@graph: Loops" begin
    g2 = @graph begin
        @nodes begin
            neurons = for i in 1:3
                IFNeuron(I_in = i)
            end
        end
        @connections begin
            neurons[1] => neurons[2], [weight = 1.0]
            neurons[2] => neurons[3], [weight = 0.1]
        end
    end
    @test neurons isa Vector{IFNeuron}

    g3 = @graph begin
        f = 0.15
        N_E = 24
        N_I = Int(ceil(N_E / 4))
        N_E_selective = Int(ceil(f * N_E))
        N_E_nonselective = N_E - 2 * N_E_selective

        w₊ = 1.7 
        w₋ = 1 - f * (w₊ - 1) / (1 - f)

        exci_scaling_factor = 1600 / N_E
        inh_scaling_factor = 400 / N_I

        @nodes begin
            exci_circuits = [LIFExciCircuit(; namespace = :g, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor) for i in 1:3] 
            inh_circuit = LIFInhCircuit(; namespace = :g, N_neurons = N_E_selective, weight = w₊, exci_scaling_factor, inh_scaling_factor)
        end
        @connections begin
            circuits = [exci_circuits; inh_circuit]
            for (c1, c2) in Iterators.product(circuits, circuits)
                c1 == c2 && continue
                w = (c2 == circuits[1] || c2 == circuits[2]) ? w₋ : 1
                c1 => c2, [weight = w]
            end
        end
    end
end

@testset "@graph!" begin
    g1 = @graph begin
        @nodes begin
            n1 = IFNeuron(I_in=1)
        end
    end
    g2 = @graph! g1 begin
        @nodes begin
            n2 = IFNeuron(I_in=2)
        end
    end
    @test n1 ∈ nodes(g2)
    @test n2 ∈ nodes(g1)
    @test g1 === g2
end


@testset "@connection: basic equation" begin
    @testset "input symbol detection" begin
        struct AnotherConnection
            weight::Float64
        end
        
        # Error if nonexistent input assigned
        @connection function (conn::AnotherConnection)(src::KuramotoOscillator, dst::KuramotoOscillator, t)
            w = conn.weight
            x₀ = src.θ
            xᵢ = dst.θ

            @equations begin
                jcn = w * sin(x₀ - xᵢ)
                jcn2 = w * sin(x₀ - xᵢ)
            end
        end
        @named k1 = KuramotoOscillator()
        @named k2 = KuramotoOscillator()
        conn = AnotherConnection(1.0)
        # Complains that jcn2 is not an input to dst
        @test_throws ArgumentError conn(to_subsystem(k1), to_subsystem(k2), 0.0)
        
    end

    @connection function (conn::MyConnection)(src::KuramotoOscillator, dst::KuramotoOscillator, t)
        w = conn.weight
        x₀ = src.θ
        xᵢ = dst.θ

        @equations begin
            jcn = w * sin(x₀ - xᵢ)
        end
    end

    p = SubsystemParams{KuramotoOscillator_NonNoisy}(ω = 249.0)
    θ1, θ2 = rand(2)

    st1 = SubsystemStates{KuramotoOscillator_NonNoisy}(θ = θ1)
    st2 = SubsystemStates{KuramotoOscillator_NonNoisy}(θ = θ2)
    k1 = GraphDynamics.Subsystem(st1, p)
    k2 = GraphDynamics.Subsystem(st2, p)
    w = rand()
    conn1 = BasicConnection(w)
    conn2 = MyConnection(w)
    @test conn1(k1, k2, 0.).jcn == conn2(k1, k2, 0.).jcn == w*sin(θ1 - θ2)
end

@testset "@connection: discrete events" begin
    @connection function (conn::MyEventConnection)(src::TAN, dst::Matrisome, t)
        t_event = conn.event_times.t_event
        w = conn.weight

        @equations begin
            jcn = w * dst.TAN_spikes
        end
        @event_times t_event
        @discrete_events begin
            (t == t_event) => (dst.TAN_spikes = w * rand(src.rng, Poisson(src.R)))
        end
    end
    
    # Error on multiple arguments to event_times
    @test_throws ErrorException @macroexpand @connection function (conn::MyEventConnection)(src::TAN, dst::Matrisome, t)
        t_event = conn.event_times.t_event
        w = conn.weight

        @equations begin
            jcn = w * dst.TAN_spikes
        end
        @event_times t_event t_event2
        @discrete_events begin
            (t == t_event) => (dst.TAN_spikes = w * rand(src.rng, Poisson(src.R)))
        end
    end

    g1 = @graph begin
        @nodes begin
            tan = TAN()
            str = Striatum()
        end
        @connections begin
            tan => str, [weight = 0.5, t_event = 1.0]
        end
    end

    g2 = GraphSystem()
    @named tan = TAN()
    @named str = Striatum()
    GraphDynamics.add_node!(g2, tan)
    GraphDynamics.add_node!(g2, str)
    GraphDynamics.add_connection!(g2, tan, str; conn = MyEventConnection(0.5, (; t_event = 1.0)), t_event = 1.0)
    compare_graph_solutions(g1, g2, (0., 2.));
end

@testset "@connection: Union types" begin
    @connection function (conn::MyConnection)(
        src::Union{HHNeuronInhib_FSI_Adam, HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam},
        dst::Union{HHNeuronInhib_MSN_Adam, HHNeuronExci_STN_Adam, HHNeuronInhib_GPe_Adam},
        t
    )
        w = conn.weight
        @equations begin
            I_syn = -w * src.G * (dst.V - src.E_syn)
        end
    end

    conn = MyConnection(3.)
    @test hasmethod(conn, (Subsystem{HHNeuronInhib_FSI_Adam}, Subsystem{HHNeuronInhib_MSN_Adam}, Any))
    @test hasmethod(conn, (Subsystem{HHNeuronExci_STN_Adam}, Subsystem{HHNeuronInhib_GPe_Adam}, Any))
end

@testset "Wiring rule + docstring" begin
    @blox struct Foo(name::Symbol, namespace, a)
        @params a
        @states x=1.0
        @inputs
        @equations begin
            D(x) = a*x
        end
    end

    """
    foo
    """
    @wiring_rule (src::Foo, dst::Foo; kwargs...) begin
        @connections begin
            @rule src => dst, [conn=PSPConnection(10.0)]
        end
    end

    @test string(@doc(system_wiring_rule!(::Foo, ::Foo))) == "foo\n"
end


