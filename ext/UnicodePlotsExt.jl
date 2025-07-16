module UnicodePlotsExt

using UnicodePlots

using Neuroblox: Neuroblox

function __init__()
    @eval function Neuroblox.GraphDynamicsInterop.maybe_show_plot(trace)
        window_size=20
        run_mean = map(eachindex(trace)) do i
            inds = max(1, i-window_size):i
            sum(i -> trace[i].iscorrect, inds)/length(inds) * 100
        end
        [("Plot", lineplot(run_mean, xlabel="Trial", ylabel="Accuracy", ylim=(0, 100))),]
    end
end

end #module UnicodePlotsExt
