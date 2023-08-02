function progress_scope(params; lvl=1)
    para_list = []
    for p in params
        pp = ModelingToolkit.unwrap(p)
        if ModelingToolkit.hasdefault(pp)
            d = ModelingToolkit.getdefault(pp)
            if typeof(d)==SymbolicUtils.BasicSymbolic{Real}
                pp = DelayParentScope(pp,lvl)
            end
        end
        push!(para_list,ModelingToolkit.wrap(pp))
    end
    return para_list
end
