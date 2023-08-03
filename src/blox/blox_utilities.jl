function progress_scope(params; lvl=0)
    para_list = []
    for p in params
        pp = ModelingToolkit.unwrap(p)
        if ModelingToolkit.hasdefault(pp)
            d = ModelingToolkit.getdefault(pp)
            if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
                if lvl==0
                    pp = ParentScope(pp)
                else
                    pp = DelayParentScope(pp,lvl)
                end
            end
        end
        push!(para_list,ModelingToolkit.wrap(pp))
    end
    return para_list
end

"""
This function progresses the scope of parameters and leaves floating point values untouched
"""
function progress_scope(args...)
    paramlist = []
    for p in args
        if p isa Float64
            push!(paramlist, p)
        else
            p = ParentScope(p)
            # pp = ModelingToolkit.unwrap(p)
            # if ModelingToolkit.hasdefault(pp)
            #     d = ModelingToolkit.getdefault(pp)
            #     if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
            #         pp = ParentScope(pp)
            #     end
            # end
            # push!(para_list,ModelingToolkit.wrap(pp))
            push!(paramlist, p)
        end
    end
    return paramlist
end

"""
    This function compiles already existing parameters with floats after making them parameters.
    Keyword arguments are used because parameter definition requires names, not just values
"""
function compileparameterlist(;kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Float64
            paramlist = vcat(paramlist, @parameters $kw = v)
        else
            paramlist = vcat(paramlist, v)
        end
    end
    return paramlist
end