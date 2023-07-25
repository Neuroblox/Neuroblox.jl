function scope_dict!(para_dict)
    Dict(typeof(v) == Num ? n => ParentScope(v) : n => (@parameters $n=v)[1] for (n,v) in para_dict)
end
