using OrdinaryDiffEqVerner, OrdinaryDiffEqRosenbrock
using CSV, DataFrames
using Random
using Neuroblox
using BenchmarkTools

const SUITE = BenchmarkTools.BenchmarkGroup()

SUITE["Create LinearNeuralMass"] = @benchmarkable lm1 = LinearNeuralMass(; name = :lm1)

##########################
### Harmonic Oscillator
##########################

harmonic_oscillator = SUITE["Harmonic Oscillator"] = BenchmarkGroup()

function make_harmonic_oscillator()
    g = @graph begin
        @nodes begin
            osc1 = HarmonicOscillator()
            osc2 = HarmonicOscillator()
        end
        @connections begin
            osc1 => osc2, [weight = 1.0]
            osc2 => osc1, [weight = 1.0]
        end
    end
end

harmonic_oscillator["System creation"] = @benchmarkable $make_harmonic_oscillator()

g = make_harmonic_oscillator()
sim_dur = 20_000.0 # 20 seconds
prob = ODEProblem(g, [], (0.0, sim_dur),[])

harmonic_oscillator["Solve"] = @benchmarkable solve($prob, AutoVern7(Rodas4()), saveat=0.1)



##########################
######################
#### 40 Random NMMs
######################

rng = Xoshiro(308)

function make_nmm_network(; rng=rng, N=40, sim_dur=50.0)
    @graph begin
        @nodes begin
            neurons = for i in 1:N
                type = rand(rng, (HarmonicOscillator, LinearNeuralMass, WilsonCowan))
                type()
            end
        end

        adj = randn(rng, N, N)
        @connections begin
            for I ∈ CartesianIndices(adj)
                if rand(rng) < 0.20
                    continue
                else
                    w = adj[I]
                    neurons[I.I[2]] => neurons[I.I[1]], [weight = w]
                end
            end
        end
    end
    prob = ODEProblem(g, [], (0.0, sim_dur),[])
end

nmms = SUITE["40 Random Neural Mass Models"] = BenchmarkGroup()
nmms["System creation"] = @benchmarkable $make_nmm_network()

prob = make_nmm_network()
nmms["Solve"] = @benchmarkable solve($prob, Vern7())

nmm400 = SUITE["400 Random Neural Mass Models"] = BenchmarkGroup()
nmm400["System creation"] = @benchmarkable $make_nmm_network(; N = 400)

prob = make_nmm_network(; N = 400)
nmm400["Solve"] = @benchmarkable solve($prob, Vern7())

###############################
# Category Learning
###############################

RF = SUITE["Category Learning"] = BenchmarkGroup()

fn = joinpath(@__DIR__(), "..", "examples", "image_example.csv")
data = CSV.read(fn, DataFrame)

function make_rf_model(; time_block_dur = 90, global_ns = :g, N_trials = 3, data = data)
    g = GraphSystem()

    #define the circuit blox
    @graph g begin
        @nodes begin
            stim = ImageStimulus(data[1:N_trials,:]; t_stimulus=600, t_pause=1000)
            LC = NextGenerationEI(;Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 
            
            VC = Cortical(N_wta=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0)

            PFC = Cortical(N_wta=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0)

            STR1 = Striatum(N_inhib=1)
            STR2 = Striatum(N_inhib=1)

            tan_pop1 = TAN() 
            tan_pop2 = TAN() 

            AS = GreedyPolicy(t_decision=180.5)
            SNcb = SNc()

        end

        #define learning rules
        hebbian_mod = HebbianModulationPlasticity(K=0.04, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=1600-eps(), t_post=1600-eps(), t_mod=90)
        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=5, t_pre=1600-eps(), t_post=1600-eps())
        hebbian_thal_cort = HebbianPlasticity(K=1.7e-5, W_lim=6, t_pre=1600-eps(), t_post=1600-eps())

        @connections begin
            LC => VC, [weight = 44] #LC->VC
            LC => PFC, [weight = 44] #LC->pfc

            VC => PFC, [weight = 1, density = 0.08, learning_rule = hebbian_cort] #VC->pfc
            PFC => STR1, [weight = 0.075, density = 0.04, learning_rule = hebbian_mod] #pfc->str1
            PFC => STR2, [weight = 0.075, density = 0.04, learning_rule = hebbian_mod] #pfc->str2
            
            stim => VC, [weight = 14] #stim->VC
            tan_pop1 => STR1, [weight = 1, t_event = 90.0] #TAN pop1 -> str1
            tan_pop2 => STR2, [weight = 1, t_event = 90.0] #TAN pop2 -> str2
            STR1 => tan_pop1, [weight = 1] #str1 -> TAN pop1 
            STR2 => tan_pop1, [weight = 1] #str2 -> TAN pop1
            STR1 => tan_pop2, [weight = 1] #str1 -> TAN pop2 
            STR2 => tan_pop2, [weight = 1] #str2 -> TAN pop2
            STR1 => STR2, [weight = 1, t_event = 181.0] #str1 -> str2
            STR2 => STR1, [weight = 1, t_event = 181.0] #str2 -> str1
            STR1 => AS  # str1->AS
            STR2 => AS  # str2->AS
            STR1 => SNcb, [weight = 1] # str1->Snc
            STR2 => SNcb, [weight = 1]  # str2->Snc
        end
    end

    agent = Agent(g; t_block = 90)
    env = ClassificationEnvironment(stim)

    return agent, env
end

RF["System creation"] = @benchmarkable $make_rf_model()

agent, env = make_rf_model();
reset!(env)

RF["Solve"] = @benchmarkable run_experiment!($agent, $env; alg=Vern7(), reltol=1e-9, abstol=1e-9)
