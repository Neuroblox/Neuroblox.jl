using Distributed
using DataFrames
using MacroTools
using Random
using DelimitedFiles
using Dates
using BenchmarkTools

Random.seed!(12345)
include(joinpath(@__DIR__(), "utils.jl"))
include(joinpath(@__DIR__(), "setup_snippets.jl"))

function run_and_save_benchmarks(benches=to_benchmark,
                                 filename=joinpath(@__DIR__(),
                                                   "benchmark_history",
                                                   "benchmarks-$(now()).csv"))
    df = summarize_benchmark_data(runbench.(benches))
    writedlm(filename, Iterators.flatten(([names(df)], eachrow(df))))
end

"""
    to_benchmark

Set of benchmarkable examples (each of which is runnable with `time_to_first_x`)
"""
to_benchmark = [
    BenchSetup(name = "Load NeuroBlox",
               bench = :(@eval using Neuroblox))
    
    BenchSetup(name = "Create LinearNeuralMass",
               prelude = :(using Neuroblox),
               bench = :(@named lm1 = LinearNeuralMass()))
    
    BenchSetup(name = "Harmonic Oscillator creation",
               prelude = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
               bench = harmonic_creation)
    
    BenchSetup(name="Harmonic Oscillator solve",
               prelude=quote
                   using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
                   $harmonic_creation
               end,
               args = [:prob],
               bench = :(sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)),
               seconds = 50)
    
    BenchSetup(name="Harmonic Oscillator with parameter weights creation",
               prelude=:(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
               bench=harmonic_with_weights_creation)
    
    BenchSetup(name="Harmonic Oscillator with parameter weights solve",
               prelude=quote
                   using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
                   $harmonic_with_weights_creation
               end,
               args = [:prob],
               bench = :(sol = solve(prob, AutoVern7(Rodas4()), saveat=0.1)),
               seconds = 50)
    
    BenchSetup(name="Jansen-Ritt creation",
               prelude = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
               bench=jansen_ritt_creation)

    BenchSetup(name="Jansen-Ritt solve",
               prelude = quote
                   using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
                   $jansen_ritt_creation
               end,
               args = [:prob],
               bench=quote
                   sol_dde_no_delays = solve(prob, MethodOfSteps(Vern7()), saveat=1)
               end)

    BenchSetup(name="80 randomly connected neurons setup",
               prelude = :(using Neuroblox, DifferentialEquations, Graphs, MetaGraphs),
               bench =  eighty_neurons)

    BenchSetup(name="80 randomly connected neurons solve",
               prelude = quote
                   using Neuroblox, DifferentialEquations, Graphs, MetaGraphs
                   $eighty_neurons
               end,
               args = [:prob,],
               bench = :(solve(prob, AutoVern7(Rodas4()), saveat=0.1)))

    BenchSetup(name="RF_learning_simple neuron agent/environment creation",
               prelude=quote
                   using CSV,DataFrames, Neuroblox, DifferentialEquations, MetaGraphs
                   
                   time_block_dur = 90 # ms (size of discrete time blocks)
                   N_trials = 1 #number of trials

                   global_ns = :g
                   
                   fn = joinpath(@__DIR__(), "..", "examples", "image_example.csv") #stimulus image file
                   data = CSV.read(fn, DataFrame)
               end,
               args = [:time_block_dur, :N_trials, :global_ns, :data],
               bench=rf_learning_setup)
    
    BenchSetup(name="RF_learning_simple run_experiment (single trial)",
               prelude=quote
                   using CSV,DataFrames, Neuroblox, DifferentialEquations, MetaGraphs
                   
                   time_block_dur = 90 # ms (size of discrete time blocks)
                   N_trials = 1 #number of trials

                   global_ns = :g

                   fn = joinpath(@__DIR__(), "..", "examples", "image_example.csv")
                   data = CSV.read(fn, DataFrame)
                   $rf_learning_setup
               end,
               setup = quote
                   reset!(agent)
                   reset!(env)
               end,
               args = [:agent, :env],
               bench=quote
                   run_experiment!(agent, env; alg=Vern7(), reltol=1e-9, abstol=1e-9)
               end,
               # This has to be 1 otherwise we'd get errors since the `setup` code only runs once
               # per benchmark-sample, not once per eval in the sample:
               evals = 1,
               seconds=5_000)

    BenchSetup(name="RF_learning_simple run_experiment (three trials)",
               prelude=quote
                   using CSV,DataFrames, Neuroblox, DifferentialEquations, MetaGraphs
                   
                   time_block_dur = 90 # ms (size of discrete time blocks)
                   N_trials = 3 #number of trials

                   global_ns = :g

                   fn = joinpath(@__DIR__(), "..", "examples", "image_example.csv")
                   data = CSV.read(fn, DataFrame)
                   $rf_learning_setup
               end,
               setup = quote
                   reset!(agent)
                   reset!(env)
               end,
               args = [:agent, :env],
               bench=quote
                   run_experiment!(agent, env; alg=Vern7(), reltol=1e-9, abstol=1e-9)
               end,
               # This has to be 1 otherwise we'd get errors since the `setup` code only runs once
               # per benchmark-sample, not once per eval in the sample:
               evals = 1,
               seconds=5_000)
]

