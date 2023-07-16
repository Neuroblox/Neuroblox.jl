function scope_dict(para_dict::Dict{Symbol, Union{Float64,Num}})
    @show para_dict
    para_dict_copy = copy(para_dict)
    for (n,v) in para_dict_copy
        if typeof(v) == Num
            para_dict_copy[n] = ParentScope(v)
        else
            @show n
            @show v
            para_dict_copy[n] = (@parameters $n=v)[1]
        end
    end
    return para_dict_copy
end
