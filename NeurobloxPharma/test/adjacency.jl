using NeurobloxPharma, OrdinaryDiffEqVerner, SparseArrays, Random
using Test, StableRNGs, CairoMakie, ReferenceTests, CSV, DataFrames

@testset "AdjacencyMatrix [HH Neurons]" begin
    @graph g begin
        @nodes begin
            n1 = HHNeuronExci()
            n2 = HHNeuronExci()
            n3 = HHNeuronInhib()
        end
        @connections begin
            n1 => n2, [weight = 1]
            n1 => n3, [weight = 1]
            n3 => n2, [weight = 1]
            n2 => n2, [weight = 1]
        end
    end
    adj = AdjacencyMatrix(g) 
    A = [0 1 1 ; 0 1 0; 0 1 0]
    @test all(A .== adj.matrix)
    @test all([:g₊n1, :g₊n2, :g₊n3] .== adj.names)
end

@testset "AdjacencyMatrix [Cortical]" begin
    global_ns = :g

    A = Matrix{Matrix{Bool}}(undef, 2, 2)
    A[2,1] = [0 1 ; 1 1]
    A[1,2] = [0 1 ; 1 1]

    @named cb1 = Cortical(namespace = global_ns, N_wta=2, N_exci=2, connection_matrices=A, weight=1)

    adj = AdjacencyMatrix(cb1) 

    adj_wta_11 = [0 1 1; 1 0 0; 1 0 0]
    adj_wta_12 = [[0 0 0]; hcat([0, 0], A[1,2])]
    adj_wta_21 = [[0 0 0]; hcat([0, 0], A[2,1])]

    A_wta = [adj_wta_11 adj_wta_12 ; adj_wta_21 adj_wta_11]

    A = [
        hcat(A_wta, [0, 0, 0, 0, 0, 0]);
        [0 1 1 0 1 1 0]
    ]

    @test sum(A) == nnz(adj.matrix)

    nms = [
        :g₊cb1₊wta1₊inh,
        :g₊cb1₊wta1₊exci1,
        :g₊cb1₊wta1₊exci2,
        :g₊cb1₊wta2₊inh,
        :g₊cb1₊wta2₊exci1,
        :g₊cb1₊wta2₊exci2,
        :g₊cb1₊ff_inh
    ]

    @test all(n -> n in nms, adj.names) && length(nms) == length(adj.names)
end

@testset "AdjacencyMatrix [Agent]" begin
    N_trials = 10 ## number of trials
    trial_dur = 1000 ## in ms

    ## download the stimulus images 
    image_set = CSV.read(joinpath(@__DIR__(), "../../examples/smaller_cs_stimuli_set.csv"), DataFrame) ## reading data into DataFrame format

    ## define stimulus Blox
    ## t_stimulus: how long the stimulus is on (in ms)
    ## t_pause : how long the stimulus is off (in ms)
    rng = StableRNG(7232189)
    @graph g begin
        @nodes begin
            stim = ImageStimulus(image_set;  t_stimulus=trial_dur, t_pause=0); 

            ## Cortical Bloxs
            VAC = Cortical(; N_wta=4, N_exci=5,  density=0.05, weight=1, rng) 
            AC = Cortical(;  N_wta=2, N_exci=5, density=0.05, weight=1, rng) 
            ## ascending system Blox, modulating frequency set to 16 Hz
            ASC1 = NextGenerationEI(; Cₑ=2*26,Cᵢ=1*26, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑᵢ=0.6*26, kᵢₑ=0.6*26)
        end
        ## learning rule
        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=15, t_pre=trial_dur, t_post=trial_dur) 
        @connections begin
            stim => VAC, [weight=14, rng=rng] 
            ASC1 => VAC, [weight=44, rng=rng]
            ASC1 => AC, [weight=44, rng=rng]
            VAC => AC, [weight=3, density=0.1, learning_rule = hebbian_cort, rng=rng] ## pass learning rule as a keyword argument
        end
    end

    agent = Agent(g)
    env = ClassificationEnvironment(stim, N_trials)

    let fig = Figure(size = (1600, 800))
        adjacency(fig[1,1], agent; title="Initial weights", colorrange=(0,7))
        run_experiment!(agent, env; t_warmup=200.0, alg=Vern7())
        adjacency(fig[1,2], agent; title="Final weights", colorrange=(0,7))

        @test_reference "plots/adj_experiment.png" fig by=psnr_equality(50)
    end
end
