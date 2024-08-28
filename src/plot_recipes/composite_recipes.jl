function MakieCore.convert_arguments(::Type{<: MakieCore.AbstractPlot}, sol::SciMLBase.AbstractSolution, cb::CompositeBlox)
    V = average_voltage_timeseries(sol, cb)

    return (sol.t, vec(V))
end