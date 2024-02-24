include(joinpath(@__DIR__(), "benchmarks.jl"))

open(joinpath(@__DIR__(), "README.md"), "w+") do io
    println(io, """
    # Neuroblox Benchmark suite

    Tools for benchmarking Neuroblox.jl
    """)
    for sym ∈ (:run_and_save_benchmarks, :BenchSetup, :neuroblox_benchmark_suite, :run, :summarize_benchmark_data)
        println(io, "<details><summary> $sym </summary>\n<p>\n")
        println(io, Base.Docs.doc(Base.Docs.Binding(@__MODULE__(), sym)))
        println(io, "\n</details>\n</p>")
        println(io, "\n____________________________\n")
    end
end
