
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
        TimeData(;# value = val,
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

Base.@kwdef struct BenchSetup
    name::String
    prelude::Expr = Expr(:block, nothing)
    setup::Expr = Expr(:block, nothing)
    args::Vector{Symbol} = Symbol[]
    bench::Expr
    teardown::Expr = Expr(:block, nothing)
    seconds::Int = 1000
    samples::Int = 100
    evals::Int = 5
end

maybe_quote(x) = x isa Union{Symbol, Expr} ? QuoteNode(x) : x

function runbench((; name, prelude, setup, args, bench, teardown, seconds, evals, samples)::BenchSetup)
    # Create a single distributed proc 
    id = only(addprocs(1))
    @info "Running benchmark \"$(name)\""
    bench_args = map(args) do arg
        :($arg = $(Expr(:$, :(maybe_quote($arg)))))
    end
    bench_setup = Expr(:block, bench_args..., setup)
    if Base.isexpr(prelude, :block)
        prelude.head = :toplevel
    end
    f = Distributed.spawnat(
        id, () -> begin
            @eval begin
                using BenchmarkTools
                include(joinpath(@__DIR__(), "utils.jl"))
            end
            @eval $prelude
            
            ttfx = @eval let $((:($arg = $arg) for arg ∈ args)...)
                $setup
                res = @eval @time_data $name $bench
                $teardown
                res
            end
            @info "TTFX result for $name: " ttfx
            
            runtime_bench = @eval let $((:($arg = $arg) for arg ∈ args)...)
                @benchmark($bench, setup=$bench_setup, teardown=$teardown,
                           seconds=$seconds, evals=$evals, samples=$samples)
            end
            @info "runtime benchmark result for $name: " runtime_bench
            (;name=name, ttfx_data=ttfx, runtime_data=runtime_bench)
        end
    )
    result = fetch(f)
    rmprocs([id])
    result
end

function summarize_benchmark_data(bench_results)
    v = map(bench_results) do (; name, ttfx_data, runtime_data)
        (;
         benchmark_name=name,
         
         initial_runtime_ns = float(ttfx_data.run_time_ns),
         initial_compiletime_percent = 100 * ttfx_data.compile_time_ns / ttfx_data.run_time_ns,
         warmed_up_minimum_runtime_ns = minimum(runtime_data.times),

         
         initial_alloced_bytes = ttfx_data.allocated_bytes,
         initial_gc_time_ns = float(ttfx_data.gc_time_ns),
         initial_gc_alloc_count = ttfx_data.gc_alloc_count,

         warmed_up_allocated_bytes = runtime_data.memory,
         warmed_up_mean_gc_time_ns = mean(runtime_data.gctimes),
         warmed_up_gc_alloc_count = runtime_data.allocs,
         )
    end
    DataFrame(v)
end

