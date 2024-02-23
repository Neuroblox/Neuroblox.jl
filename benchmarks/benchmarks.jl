using Distributed
using DataFrames
using MacroTools
using Random
using DelimitedFiles
using Dates
using BenchmarkTools
using CSV

seed = 12345
Random.seed!(seed)
include(joinpath(@__DIR__(), "utils.jl"))
include(joinpath(@__DIR__(), "benchmarkable_examples.jl"))

"""
    
"""
function run_and_save_benchmarks(benches=to_benchmark,
                                 filename=joinpath(@__DIR__(),
                                                   "benchmark_history",
                                                   "benchmarks-$(now()).csv"))
    df = DataFrame(summarize_benchmark_data.(run.(benches)))
    writedlm(filename, Iterators.flatten(([names(df)], eachrow(df))))
end

"""
    to_benchmark

Set of benchmarkable examples (each of which is runnable with `time_to_first_x`)
"""
to_benchmark = [
    load_neuroblox
    linear_neural_mass_creation
    
    harmonic_oscillator_system_creation
    harmonic_oscillator_structural_simplify
    harmonic_oscillator_solve
    
    jansen_ritt_system_creation
    jansen_ritt_structural_simplify
    jansen_ritt_solve

    eighty_neurons_system_creation
    eighty_neurons_structural_simplify
    eighty_neurons_solve

    rf_learning_setup
    rf_learning_three_trials
]

