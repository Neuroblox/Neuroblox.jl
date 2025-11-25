using
    GraphDynamics,
    Test,
    Distributions,
    ModelingToolkit,
    Random,
    StochasticDiffEq,
    NeurobloxBasics,
    LinearAlgebra

using GraphDynamics.OhMyThreads:
    OhMyThreads,
    SerialScheduler,
    StaticScheduler,
    DynamicScheduler

using NeurobloxBase: AbstractNeuralMass, AbstractNeuron
using Base.Iterators: map as imap
using GraphDynamics.SymbolicIndexingInterface

using ForwardDiff: ForwardDiff
using FiniteDiff: FiniteDiff
using DiffEqCallbacks: DiffEqCallbacks, PeriodicCallback

using NeurobloxBase.GraphDynamicsInterop: t_block_event

using SciMLBase: successful_retcode, solve

function test_compare_du_and_sols(::Type{ODEProblem}, g, tspan;
                                  u0map=[], param_map=[],
                                  rtol,
                                  parallel=true, mtk=true, alg=nothing, params_to_compare=[],
                                  t_block=missing,
                                  save_everystep=false,
                                  cse=true)
    if g isa Tuple
        (gl, gr) = g
    else
        gl = deepcopy(g)
        gr = deepcopy(g)
    end
    @named gsys = system_from_graph(gl; graphdynamics=true)
    state_names = variable_symbols(gsys)
    sol_grp, du_grp, sol_grp_obj = let sys = gsys
        if !ismissing(t_block)
            global_events = [
                PeriodicCallback(t_block_event(:t_block_early), t_block - √(eps(float(t_block)))),
                PeriodicCallback(t_block_event(:t_block_late), t_block  +2*√(eps(float(t_block))))
            ]
        else
            global_events = []
        end
            
        prob = ODEProblem(sys, u0map, tspan, param_map; global_events)
        (; f, u0, p) = prob
        du = similar(u0)
        f(du, u0, p, 1.0)

        sol = solve(prob, alg; save_everystep)
        @test successful_retcode(sol)
        sol_u_reordered = map(state_names) do name
            sol[name][end]
        end
        du_reordered = map(state_names) do name
            getu(sys, name)(du)
        end
        sol_u_reordered, du_reordered, sol
    end
   
    if mtk
        sol_mtk, du_mtk, sol_mtk_obj = let @named sys = system_from_graph(gr; graphdynamics=false, t_block)
            prob = ODEProblem(sys, u0map, tspan, param_map; cse)
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)
            # display(f.f.f_iip)
            sol = solve(prob, alg; save_everystep)
            @test successful_retcode(sol)
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            # For some reason getu is erroring here, this is some sort of MTK bug I think
            # du_reordered = map(state_names) do name 
            #     getu(sys, name)(du)
            # end
            du_reordered = du
            sol_u_reordered, du_reordered, sol
        end
        for i ∈ eachindex(state_names)
            if !isapprox(sol_grp[i], sol_mtk[i]; rtol=rtol)
                @debug  "" i state_names[i] sol_grp[i] sol_mtk[i]
            end
        end
        @debug "" norm(sol_grp .- sol_mtk) / norm(sol_mtk)
        @test sort(du_grp) ≈ sort(du_mtk) # due to the MTK getu bug, we'll compare the sorted versions
        @test sol_grp ≈ sol_mtk rtol=rtol
        for name ∈ params_to_compare
            @debug name last(getsym(sol_grp_obj, name)(sol_grp_obj)) (getsym(sol_mtk_obj, name)(sol_mtk_obj))
            @test last(getsym(sol_grp_obj, name)(sol_grp_obj)) ≈ (getsym(sol_mtk_obj, name)(sol_mtk_obj)) rtol=rtol
        end
    end
    if parallel
        sol_grp_p, du_grp_p, sol_grp_p_obj = let sys = gsys
            if !ismissing(t_block)
                global_events = [
                    PeriodicCallback(t_block_event(:t_block_early), t_block - √(eps(float(t_block)))),
                    PeriodicCallback(t_block_event(:t_block_late), t_block  +2*√(eps(float(t_block))))
                ]
            else
                global_events = []
            end
            prob = ODEProblem(sys, u0map, tspan, param_map; scheduler=StaticScheduler(), global_events)
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)

            sol = solve(prob, alg; save_everystep)
            @test successful_retcode(sol)
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            du_reordered = map(state_names) do name
                getu(sys, name)(du)
            end
            sol_u_reordered, du_reordered, sol
        end
        @test du_grp ≈ du_grp_p
        @test sol_grp ≈ sol_grp_p rtol=rtol
        for name ∈ params_to_compare
            @test last(getsym(sol_grp_obj, name)(sol_grp_obj)) ≈ last(getsym(sol_grp_p_obj, name)(sol_grp_p_obj)) rtol=rtol
        end
    end
end

function test_compare_du_and_sols(::Type{SDEProblem}, graph, tspan; rtol, mtk=true, alg=nothing, seed=1234,
                                  u0map=[], param_map=[],
                                  sol_comparison_broken=false, f_comparison_broken=false, g_comparison_broken=false)
    Random.seed!(seed)
    if graph isa Tuple
        (graph_l, graph_r) = graph
    else
        graph_l = graph
        graph_r = graph
    end
    @named gsys = system_from_graph(graph_l; graphdynamics=true)
    state_names = variable_symbols(gsys)
    sol_grp, du_grp, dnoise_grp = let sys = gsys
        prob = SDEProblem(sys, u0map, tspan, param_map, seed=seed)
        (; f, g, u0, p) = prob
        du = similar(u0)
        f(du, u0, p, 1.1)
        dnoise = zero(u0)
        g(dnoise, u0, p, 1.1)

        @test successful_retcode(solve(prob, ImplicitEM(), saveat = 0.01,reltol=1e-4,abstol=1e-4))
        
        sol = solve(prob, alg, saveat = 0.01)
        @test successful_retcode(sol)
        sol_reordered = map(state_names) do name
            sol[name][end]
        end
        sol_reordered, collect(du), collect(dnoise)
    end
    if mtk
        sol_mtk, du_mtk, dnoise_mtk = let neuron_net = system_from_graph(graph_r; name=:neuron_net)
            prob = SDEProblem(neuron_net, u0map, tspan, param_map, seed=seed)
            (; f, g, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.1)

            dnoise = g(u0, p, 1.1)
            dnoise = sum(dnoise; dims=2)[:] # MTK might not understand that the noise is diagonal, so it can give a diagonal matrix instead
            sol = solve(prob, alg, saveat = 0.01)
            @test successful_retcode(sol)
            sol_reordered = map(state_names) do name
                sol[name][end]
            end
            sol_reordered, collect(du), collect(dnoise)
        end
        @debug "" norm(sol_grp .- sol_mtk) / norm(sol_grp)
        #due to the MTK getu bug, we'll compare the sorted versions
        @test sort(du_grp) ≈ sort(du_mtk)         broken=f_comparison_broken   
        @test sort(dnoise_grp) ≈ sort(dnoise_mtk) broken=g_comparison_broken
        @test sol_grp ≈ sol_mtk rtol=rtol         broken=sol_comparison_broken
    end
    nothing
end


function test_compare_du_and_sols_ensemble(::Type{SDEProblem}, graph, tspan; rtol, mtk=true, alg=nothing, trajectories=100_000)
    # Random.seed!(1234)
    if graph isa Tuple
        (graph_l, graph_r) = graph
    else
        graph_l = graph
        graph_r = graph
    end
    
    @named gsys = system_from_graph(graph_l; graphdynamics=true)
    state_names = variable_symbols(gsys)
    
    sol_grp_ens, du_grp, dnoise_grp = let sys = gsys
        prob = SDEProblem(sys, [], tspan, [])
        (; f, g, u0, p, noise_rate_prototype) = prob
        du = similar(u0)
        f(du, u0, p, 1.1)
        dnoise = zero(u0)
        g(dnoise, u0, p, 1.1)

        ens_prob = EnsembleProblem(prob)
        sols = solve(ens_prob, alg, EnsembleThreads(); trajectories)

        n_success = 0
        for sol ∈ sols
            n_success += successful_retcode(sol)
        end
        
        @test n_success > 0.95 * trajectories # allow up to 5% of the trajectories to fail
        sol_succ = [sol for sol in sols if successful_retcode(sol)]
        sols_u_reordered = map(sol_succ) do sol
            map(state_names) do name
                sol[name][end]
            end
        end
        sols_u_reordered, collect(du), collect(dnoise)
    end
    if mtk
        sol_mtk_ens, du_mtk, dnoise_mtk = let neuron_net = system_from_graph(graph_r; name=:neuron_net)
            prob = SDEProblem(neuron_net, [], tspan, [])
            (; f, g, u0, p, noise_rate_prototype) = prob
            du = similar(u0)
            f(du, u0, p, 1.1)

            dnoise = g(u0, p, 1.1)
            ens_prob = EnsembleProblem(prob)
            sols = solve(ens_prob, alg, EnsembleThreads(); trajectories)

            n_success = 0
            for sol ∈ sols
                n_success += successful_retcode(sol)
            end
            @test n_success > 0.95 * trajectories # allow up to 5% of the trajectories to fail
            sol_succ = [sol for sol in sols if successful_retcode(sol)]
            sols_u_reordered = map(sol_succ) do sol
                map(state_names) do name
                    sol[name][end]
                end
            end
            dnoise = sum(dnoise; dims=2)[:] # MTK doesn't understand that the noise is diagonal, so has a noise matrix instead
            sols_u_reordered, du, dnoise
        end
        @test sort(du_grp) ≈ sort(du_mtk)         #due to the MTK getu bug, we'll compare the sorted versions
        @test sort(dnoise_grp) ≈ sort(dnoise_mtk) #due to the MTK getu bug, we'll compare the sorted versions
        @debug "" norm(mean(sol_grp_ens) .- mean(sol_mtk_ens)) / norm(mean(sol_grp_ens))
        @test mean(sol_grp_ens) ≈ mean(sol_mtk_ens) rtol=rtol
        @test std(sol_grp_ens)  ≈ std(sol_mtk_ens)  rtol=rtol
    end
    nothing
end

function test_jacobian(f, v; rtol=1e-3)
    jac_autodiff   = ForwardDiff.jacobian(f, v)
    jac_finitediff = FiniteDiff.finite_difference_jacobian(f, v)
    @test jac_autodiff ≈ jac_finitediff rtol=rtol
end

function test_derivative(f, v; rtol=1e-3)
    d_autodiff   = ForwardDiff.derivative(f, v)
    d_finitediff = FiniteDiff.finite_difference_derivative(f, v)
    @test d_autodiff ≈ d_finitediff rtol=rtol
end
