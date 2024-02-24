# Neuroblox Benchmark suite

Tools for benchmarking Neuroblox.jl

<details><summary> run_and_save_benchmarks </summary>
<p>

```
run_and_save_benchmarks(; benches=neuroblox_benchmark_suite,
                          filename="benchmark_history/benchmarks-$(now()).csv")
```

Run the specified list of benchmarks `benches` (default to the full benchmark suite), and save the outcomes in CSV form to the file path specified by `filename`.

The tabular benchmark data will include the following columns:

  * `benchmark_name`: a list of strings containing the names of each benchmark
  * `initial_runtime_ns`: a floating point list containing how long the first call to the benchmark function took in nanoseconds. This should be representative of a users experience in a fresh julia session.
  * `initial_compiletime_percent`: a floating point list containing percentages (0 to 100) of how much of the initial runtime was taken up by compilation.
  * `warmed_up_minimum_runtim_ns`: a floating point list describing the minimum amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.
  * `warmed_up_median_runtim_ns`: a floating point list describing the median amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.
  * `initial_alloced_bytes`: an integer list describing how many bytes were allocated while running the initial benchmark function (including compilation).
  * `initial_gc_time_ns`: a floating point list describing how long the initial benchmark function spent performing garbage collection (in nanoseconds).
  * `intial_gc_alloc_count`: an integer list containing the number of separate allocations in the initial call to the benchmark function, including allocations from compilation.
  * `warmed_up_allocated_bytes`: an integer list describing how many bytes were allocated while running the benchmark function in a `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of bytes which may be misleading for non-uniform benchmarks.
  * `warmed_up_median_gc_time_ns`: a floating point list describing the median time spent (in nanoseconds) performing garbage collection during the `BenchmarkTools.@benchmark` loop.
  * `warmed_up_gc_alloc_count`: an integer list describing how many bytes were allocated while running the benchmark function during the `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of allocations which may be misleading for non-uniform benchmarks.
  * `metadata` a list of metadata about the benchmark process. All empty except for the first entry which contains information about the latest commit of Neuroblox.jl when the benchmark was run, and detailed information about the computer which ran the benchmark.


</details>
</p>

____________________________

<details><summary> BenchSetup </summary>
<p>

```
BenchSetup(;
    name::String,
    setup::Expr = quote end,
    args::Vector{Symbol} = Symbol[],
    sample_setup::Expr = quote end,
    bench::Expr,
    sample_teardown::Expr = quote end,
    samples::Int = 100,
    evals::Int = 5,
    seconds::Int = 1000,
)
```

Create a benchmark object to be evaluated with `run`.

Required arguments:

  * `name`: the name of the benchmark to be run.
  * `bench`: an `Expr` object containing the quoted code to be benchmarked. This will be run once in a `let` block to collect compilation information, and then again inside a `BenchmarkTools.@benchmark` loop to collect warmed up runtime information.

Optional arguments:

  * `setup`: any initial, top-level setup code that needs to be run before the benchmarking begins. This code is only executed once and is intended for things like loading packages and defining functions and datastructures.
  * `args`: arguments to be passed to the `bench` expression in the benchmark loop, rather than treating them as global variables, they will be passed like function arguments to the `sample_setup` code.
  * `sample_setup`: This code is run before each benchmark sample and is intended for things like reseting mutable state before a benchmark occurs (this corresponds to the `setup` argument to `BenchmarkTools.@benchmark`).
  * `sample_teardown`: This code is run after each benchmark sample and is intended for things like reseting mutable state after a benchmark occurs (this corresponds to the `teardown` argument to `BenchmarkTools.@benchmark`).
  * `samples`: The number of samples of the runtime of `bench` to be collected by `BenchmarkTools.@benchmark`.
  * `evals`: The number of times the `bench` code should be run per-sample.
  * `seconds`: a budget of time given to `BenchmarkTools.@benchmark`. The benchmark will run either until it collects the requested number of samples, or until the time budget is exceeded.


</details>
</p>

____________________________

<details><summary> neuroblox_benchmark_suite </summary>
<p>

```
neuroblox_benchmark_suite :: Vector{BenchSetup}
```

Vector of the default Neuroblox benchmarks, see `benchmarks/benchmarkable_expamples.jl` for details.


</details>
</p>

____________________________

<details><summary> run </summary>
<p>

```
run(command, args...; wait::Bool = true)
```

Run a command object, constructed with backticks (see the [Running External Programs](@ref) section in the manual). Throws an error if anything goes wrong, including the process exiting with a non-zero status (when `wait` is true).

The `args...` allow you to pass through file descriptors to the command, and are ordered like regular unix file descriptors (eg `stdin, stdout, stderr, FD(3), FD(4)...`).

If `wait` is false, the process runs asynchronously. You can later wait for it and check its exit status by calling `success` on the returned process object.

When `wait` is false, the process' I/O streams are directed to `devnull`. When `wait` is true, I/O streams are shared with the parent process. Use [`pipeline`](@ref) to control I/O redirection.

```
run(b::BenchSetup) :: @NamedTuple{name::String, ttfx_data::TimeData, runtime_data::BenchmarkTools.Trial}
```

Take a given `BenchSetup` and run it in a distributed process generating information describing both the compile times associated with the benchmark, and the warmed-up runtimes.

This returns a `NamedTuple` of low-level data collected during the benchmark. It can be processed into something more digestable by `summarize_benchmar_data`.

```
run(b::Benchmark[, p::Parameters = b.params]; kwargs...)
```

Run the benchmark defined by [`@benchmarkable`](@ref).

```
run(group::BenchmarkGroup[, args...]; verbose::Bool = false, pad = "", kwargs...)
```

Run the benchmark group, with benchmark parameters set to `group`'s by default.


</details>
</p>

____________________________

<details><summary> summarize_benchmark_data </summary>
<p>

```
summarize_benchmark_data(bench_result::@NamedTuple{name::String, ttfx_data::TimeData, runtime_data::Trial})
```

Take low-level data returned by `run(b::BenchSetup)` and summarize it into a named tuple containing

  * `benchmark_name`: a list of strings containing the names of each benchmark
  * `initial_runtime_ns`: a floating point list containing how long the first call to the benchmark function took in nanoseconds. This should be representative of a users experience in a fresh julia session.
  * `initial_compiletime_percent`: a floating point list containing percentages (0 to 100) of how much of the initial runtime was taken up by compilation.
  * `warmed_up_minimum_runtim_ns`: a floating point list describing the minimum amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.
  * `warmed_up_median_runtim_ns`: a floating point list describing the median amount of nanoseconds each benchmark took to run in a `BenchmarkTools.@benchmark` loop. This should be representative of the actual performance of the package once compilation is out of the way.
  * `initial_alloced_bytes`: an integer list describing how many bytes were allocated while running the initial benchmark function (including compilation).
  * `initial_gc_time_ns`: a floating point list describing how long the initial benchmark function spent performing garbage collection (in nanoseconds).
  * `intial_gc_alloc_count`: an integer list containing the number of separate allocations in the initial call to the benchmark function, including allocations from compilation.
  * `warmed_up_allocated_bytes`: an integer list describing how many bytes were allocated while running the benchmark function in a `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of bytes which may be misleading for non-uniform benchmarks.
  * `warmed_up_median_gc_time_ns`: a floating point list describing the median time spent (in nanoseconds) performing garbage collection during the `BenchmarkTools.@benchmark` loop.
  * `warmed_up_gc_alloc_count`: an integer list describing how many bytes were allocated while running the benchmark function during the `BenchmarkTools.@benchmark` loop. Due to how BenchmarkTools.jl works, this currently only reports the minimum number of allocations which may be misleading for non-uniform benchmarks.


</details>
</p>

____________________________

