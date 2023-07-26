function scope_dict!(para_dict)
    Para_dict(typeof(v) == Num ? n => ParentScope(v) : n => (@parameters $n=v)[1] for (n,v) in para_dict)
end
