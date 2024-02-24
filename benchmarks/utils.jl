"""
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

Create a benchmark object to be evaluated with `run`.

Required arguments:
+ `name`: the name of the benchmark to be run.
+ `bench`: an `Expr` object containing the quoted code to be benchmarked. This will be run once in a `let` block to collect compilation information, and then again inside a `BenchmarkTools.@benchmark` loop to collect warmed up runtime information.

Optional arguments:
+ `setup`: any initial, top-level setup code that needs to be run before the benchmarking begins. This code is only executed once and is intended for things like loading packages and defining functions and datastructures.
+ `args`: arguments to be passed to the `bench` expression in the benchmark loop, rather than treating them as global variables, they will be passed like function arguments to the `sample_setup` code.
+ `sample_setup`: This code is run before each benchmark sample and is intended for things like reseting mutable state before a benchmark occurs (this corresponds to the `setup` argument to `BenchmarkTools.@benchmark`).
+ `sample_teardown`: This code is run after each benchmark sample and is intended for things like reseting mutable state after a benchmark occurs (this corresponds to the `teardown` argument to `BenchmarkTools.@benchmark`).
+ `samples`: The number of samples of the runtime of `bench` to be collected by `BenchmarkTools.@benchmark`.
+ `evals`: The number of times the `bench` code should be run per-sample. 
+ `seconds`: a budget of time given to `BenchmarkTools.@benchmark`. The benchmark will run either until it collects the requested number of samples, or until the time budget is exceeded.
"""
Base.@kwdef struct BenchSetup
    name::String
    setup::Expr = Expr(:block, nothing)
    sample_setup::Expr = Expr(:block, nothing)
    args::Vector{Symbol} = Symbol[]
    bench::Expr
    sample_teardown::Expr = Expr(:block, nothing)
    seconds::Int = 1000
    samples::Int = 100
    evals::Int = 5
end


"""
    @time_data [name] expr

Basically just `Base.@time` copied from julia/base/timing.jl (v1.10), but modified so we
can get out the compilation time as a number (in addition to some other stuff), and attach
an optional name to the result.

This unfortunately needs to use some internals from Base.
"""
macro time_data(args...)
    if length(args) == 1
        name = ""
        ex = only(args)
    elseif length(args) == 2
        name, ex = args
        if name isa Union{Symbol, Expr}
            name = esc(name)
        end
    else
        error(ArgumentError("@time_data only accepts one or two arguments."))
    end
    quote
        ## Base uses this option to sneakily try to avoid people counting compile times, it asks the
        ## compiler to hoist compilation out of the timing block:
        # Base.Experimental.@force_compile
        ##
        local name = $name
        local gc_stats = Base.gc_num()
        local elapsedtime = Base.time_ns()
        Base.cumulative_compile_timing(true)
        local compile_elapsedtimes = Base.cumulative_compile_time_ns()
        local val = Base.@__tryfinally($(esc(ex)),
            (elapsedtime = time_ns() - elapsedtime;
             Base.cumulative_compile_timing(false);
             compile_elapsedtimes = Base.cumulative_compile_time_ns() .- compile_elapsedtimes)
                                       )
        local gc_diff = Base.GC_Diff(Base.gc_num(), gc_stats)
        TimeData(;
                 name = name,
                 run_time_ns = Int(elapsedtime),
                 compile_time_ns = Int(compile_elapsedtimes[1]),
                 recompilation_time_ns = Int(compile_elapsedtimes[2]),
                 allocated_bytes = gc_diff.allocd,
                 gc_time_ns = gc_diff.total_time,
                 gc_alloc_count = Base.gc_alloc_count(gc_diff),
                 gc_stats=gc_diff)
    end
end

Base.@kwdef struct TimeData
    name::String=""
    run_time_ns::Int
    compile_time_ns::Int
    recompilation_time_ns::Int
    allocated_bytes::Int
    gc_time_ns::Int
    gc_alloc_count::Int
    gc_stats::Base.GC_Diff
end

function Base.show(io::IO, t::TimeData)
    print(io, ("TimeData(name=\"$(t.name)\","))
    Base.time_print(io, t.run_time_ns, t.allocated_bytes, t.gc_time_ns, t.gc_alloc_count,
                    t.compile_time_ns, t.recompilation_time_ns)
    print(io, ')')
end

macro ttfx_data(name, setup, bench_code)
    quote 
        (; name= $name, setup_code= $(QuoteNode(setup)), bench_code=$(QuoteNode(bench_code)))
    end |> esc
end




maybe_quote(x) = x isa Union{Symbol, Expr} ? QuoteNode(x) : x

"""
    run(b::BenchSetup) :: @NamedTuple{name::String, ttfx_data::TimeData, runtime_data::BenchmarkTools.Trial}

Take a given `BenchSetup` and run it in a distributed process generating information describing both the compile
times associated with the benchmark, and the warmed-up runtimes.

This returns a `NamedTuple` of low-level data collected during the benchmark. It can be processed into something
more digestable by `summarize_benchmar_data`.
"""
function BenchmarkTools.run((;
                             name,
                             setup,
                             sample_setup,
                             args,
                             bench,
                             sample_teardown,
                             seconds,
                             evals,
                             samples)::BenchSetup)    
    @info "Running benchmark \"$(name)\""
    
    bench_args = map(args) do arg
        # Benchmark tools treats symbol and expr arguments in a funny way when they get interpolated,
        # so we need to insert this `maybe_quote` function to catch those.
        :($arg = $(Expr(:$, :(maybe_quote($arg)))))
    end

    # insert all these interpolated arguments into the bench setup block.
    sample_setup = Expr(:block, bench_args..., sample_setup)
    
    # This helps with if someone does `using Foo` in the `setup` block. 
    if Base.isexpr(setup, :block)
        setup.head = :toplevel
    end

    # Run the benchmarks in a fresh process from Distributed.jl. This helps ensure that the benchmark is run "fresh"
    # with no compilation from previously run benchmarks interfering with the gathered data.
    id = only(addprocs(1))
    f = Distributed.spawnat(id, () -> begin
        @eval begin
            using BenchmarkTools
            include(joinpath(@__DIR__(), "utils.jl")) # We need this for the structs and macros defined here.
        end
        @eval $setup # First run the setup code

        # Now gather the time-to-first-X benchmark information, which should include all the
        # compile times.
        ttfx = @eval let $((:($arg = $arg) for arg ∈ args)...) # Arguments that should be taken from the setup phase and passed like function arguments to the inner benchmark, rather than global variables
            
            $sample_setup # per-sample setup step (this is things like resetting mutable state)
            res = @eval @time_data $name $bench
            $sample_teardown # per-sample teardown step (this is things like resetting mutable state)
            res
        end
        @info "TTFX result for $name: " ttfx

        # Now that things have compiled, we can try to collect compiled runtime statistics with
        # BenchmarkTools.jl
        runtime_bench = @eval let $((:($arg = $arg) for arg ∈ args)...) # Arguments that should be taken from the setup phase and passed like function arguments to the inner benchmark, rather than global variables
            @benchmark($bench,
                       setup=$bench_setup,
                       teardown=$sample_teardown,
                       seconds=$seconds,
                       evals=$evals,
                       samples=$samples)
        end
        @info "runtime benchmark result for $name: " runtime_bench
        (;name=name, ttfx_data=ttfx, runtime_data=runtime_bench)
    end)
    result = fetch(f) # fetch the results
    rmprocs([id]) # throw away the process

    result
end

"""
    summarize_benchmark_data(bench_result::@NamedTuple{name::String, ttfx_data::TimeData, runtime_data::Trial})

Take low-level data returned by `run(b::BenchSetup)` and summarize it into a named tuple containing

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
"""
function summarize_benchmark_data(bench_result)
    (; name, ttfx_data, runtime_data) = bench_result 
   
    (;
     benchmark_name=name,
     
     initial_runtime_ns = float(ttfx_data.run_time_ns),
     initial_compiletime_percent = 100 * ttfx_data.compile_time_ns / ttfx_data.run_time_ns,
     warmed_up_minimum_runtime_ns = minimum(runtime_data.times),
     warmed_up_median_runtime_ns  = median(runtime_data.times),
     
     initial_alloced_bytes = ttfx_data.allocated_bytes,
     initial_gc_time_ns = float(ttfx_data.gc_time_ns),
     initial_gc_alloc_count = ttfx_data.gc_alloc_count,

     warmed_up_allocated_bytes = runtime_data.memory,
     warmed_up_median_gc_time_ns = median(runtime_data.gctimes),
     warmed_up_gc_alloc_count = runtime_data.allocs,
     )
end

