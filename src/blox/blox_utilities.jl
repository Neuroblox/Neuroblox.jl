function parameter_list(para_dict)
    params = []
    for (n,v) in para_dict
         if typeof(v) == Num
            push!(params, ParentScope(v))
        else
            push!(params, (@eval @parameters $(Meta.parse("$(n)=$(v)")))[1])
        end
    end
    return params
end
