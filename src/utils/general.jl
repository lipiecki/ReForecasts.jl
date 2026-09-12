function isunique(X::AbstractVector{<:Integer})
    iterated = Set{eltype(X)}()
    for x in X
        if x in iterated
            return false
        else
            push!(iterated, x)
        end
    end
    return true
end
