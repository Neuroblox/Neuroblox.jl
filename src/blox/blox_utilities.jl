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
