function _init(::Val{:ols}, recmatrix::AbstractMatrix{T}, intercept::AbstractVector{T}, hierarchy::Hierarchy{T}) where {T<:ValueType}
    nall, nbottom = size(hierarchy.S)

    G = Matrix{T}(undef, nbottom, nbottom)
    A = Matrix{T}(undef, nbottom, nall)

    mul!(G, hierarchy.S', hierarchy.S)
    transpose!(A, hierarchy.S)

    ldiv!(cholesky!(Symmetric(G)), A)
    
    mul!(recmatrix, hierarchy.S, A)
    fill!(intercept, zero(T))

    return nothing
end

_fit(::Val{:ols}, m::LinearReconciler, pred::AbstractMatrix, obs::AbstractMatrix, ::Vararg)::Nothing = nothing
