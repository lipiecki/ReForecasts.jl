function _init(::Val{:bu}, recmatrix::AbstractMatrix{T}, intercept::AbstractVector{T}, hierarchy::Hierarchy{T}) where {T<:ValueType}
    _, nbottom = size(hierarchy)
    fill!(recmatrix, 0)
    recmatrix[:, 1:nbottom] .= hierarchy.S 
    fill!(intercept, 0)
    return nothing
end

_fit(::Val{:bu}, m::LinearReconciler, pred::AbstractMatrix, obs::AbstractMatrix, ::Vararg)::Nothing = nothing
