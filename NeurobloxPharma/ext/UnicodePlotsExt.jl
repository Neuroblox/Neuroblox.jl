module UnicodePlotsExt

using UnicodePlots

using NeurobloxPharma: NeurobloxPharma

function __init__()
    @eval function NeurobloxPharma.maybe_show_plot(trace)
        window_size=20
        run_mean = map(eachindex(trace.trial)) do i
            inds = max(1, i-window_size):i
            sum(i -> trace.correct[i], inds)/length(inds) * 100
        end
        [("Plot", lineplot(run_mean, xlabel="Trial", ylabel="Accuracy", ylim=(0, 100))),]
    end
end

end #module UnicodePlotsExt
