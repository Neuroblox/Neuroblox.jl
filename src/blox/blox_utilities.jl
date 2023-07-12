function parameter_list(para_list)
    params = []
    for p in para_list
        if typeof(p) == Num
            push!(params, ParentScope(p))
        else
            push!(params, (@parameters p=p)[1])
        end
    end
    return params
end
