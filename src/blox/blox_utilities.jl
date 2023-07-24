Para_dict = Dict{Symbol,Union{Real,Num}}

function scope_dict!(para_dict::Para_dict)
    Para_dict(typeof(v) == Num ? n => v : n => (@parameters $n=v)[1] for (n,v) in para_dict)
end
