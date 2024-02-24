using Distributed
using DataFrames
using CSV
using MacroTools
using Random
using Dates
using BenchmarkTools
using LibGit2

seed = 12345
Random.seed!(seed)
include(joinpath(@__DIR__(), "utils.jl"))
include(joinpath(@__DIR__(), "benchmarkable_examples.jl"))

"""
    run_and_save_benchmarks(; benches=neuroblox_benchmark_suite,
                              filename="benchmark_history/benchmarks-\$(now()).csv")

Run the specified list of benchmarks `benches` (default to the full benchmark suite), and save the outcomes
in CSV form to the file path specified by `filename`.

The tabular benchmark data will include the following columns:
- `benchmark_name`: a list of strings containing the names of each benchmark
- `initial_runtime_ns`: a floating point list containing how long the first call to the benchmark function took in nanoseconds. This should be representative of a users experience in a fresh julia session.
- `initial_compiletime_percent`: a floating point list containing percentages (0 to 100) of how much of the initial runtime was taken up by compilation.
- `warmed_up_minimum_runtim_ns`: a floating point list describing the minimum amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.
- `warmed_up_median_runtim_ns`: a floating point list describing the median amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.

- `initial_alloced_bytes`: an integer list describing how many bytes were allocated while running the initial benchmark function (including compilation).
- `initial_gc_time_ns`: a floating point list describing how long the initial benchmark function spent performing garbage collection (in nanoseconds).
- `intial_gc_alloc_count`: an integer list containing the number of separate allocations in the initial call to the benchmark function, including allocations from compilation.

- `warmed_up_allocated_bytes`: an integer list describing how many bytes were allocated while running the benchmark function in a `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of bytes which may be misleading for non-uniform benchmarks.
- `warmed_up_median_gc_time_ns`: a floating point list describing the median time spent (in nanoseconds) performing garbage collection during the `BenchmarkTools.@benchmark` loop.
- `warmed_up_gc_alloc_count`: an integer list describing how many bytes were allocated while running the benchmark function during the `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of allocations which may be misleading for non-uniform benchmarks.

- `metadata` a list of metadata about the benchmark process. All empty except for the first entry which contains information about the latest commit of Neuroblox.jl when the benchmark was run, and detailed information about the computer which ran the benchmark.
"""
function run_and_save_benchmarks(; benches::AbstractVector{BenchSetup} = to_benchmark,
                                 filename  = joinpath(@__DIR__(),
                                                   "benchmark_history",
                                                   "benchmarks-$(now()).csv"))
    df = DataFrame(summarize_benchmark_data.(run.(benches)))
    df.metadata = ["" for _ ∈ 1:length(benches)]

    commit_info = let nb_path = joinpath(@__DIR__(), "..")
        LibGit2.peel(LibGit2.GitCommit, LibGit2.head(LibGit2.GitRepo(nb_path)))
    end
    system_info = let io = IOBuffer()
        versioninfo(io, ;verbose=true)
        String(take!(io))
    end

    df.metadata[1] = """
    ------------------------------
    Neuroblox Commit:

    $commit_info

    ------------------------------
    ------------------------------
    Benchmark run on:

    $system_info
    """
    CSV.write(filename, df)
end

"""
    neuroblox_benchmark_suite :: Vector{BenchSetup}

Vector of the default Neuroblox benchmarks, see `benchmarks/benchmarkable_expamples.jl` for details.
"""
neuroblox_benchmark_suite = [
    load_neuroblox
    linear_neural_mass_creation
    
    harmonic_oscillator_system_creation
    harmonic_oscillator_structural_simplify
    harmonic_oscillator_solve
    
    jansen_ritt_system_creation
    jansen_ritt_structural_simplify
    jansen_ritt_solve

    forty_neurons_system_creation
    forty_neurons_structural_simplify
    forty_neurons_solve

    rf_learning_setup
    rf_learning_three_trials
]

