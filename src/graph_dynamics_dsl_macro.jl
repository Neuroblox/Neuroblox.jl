macro neuroblox_model(fdef)
    if fdef.head != :function
        error("something informative")
    end
    sig, body = fdef.args
    for ex ∈ body.args
        
    end
end
