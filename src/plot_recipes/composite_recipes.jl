function MakieCore.convert_arguments(::MakieCore.PointBased, sol::SciMLBase.AbstractSolution, cb::CompositeBlox)
    V = average_voltage_timeseries(sol, cb)
    
    return (sol.t, vec(V))
end