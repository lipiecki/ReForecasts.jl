abstract type CombReconciler end

struct LinearCombReconciler{T<:ValueType} <: CombReconciler
    method::Val
    recmatrix::AbstractMatrix{T}
    intercept::AbstractVector{T}

    function LinearCombReconciler(A::AbstractMatrix{T}, a::AbstractVector{T}, method::Symbol) where T<:ValueType
        new{T}(Val(method), A, a)
    end

    function LinearCombReconciler(method::Symbol, hierarchy::Hierarchy{T}, ensemble::Integer) where T<:ValueType
        all = size(hierarchy)[begin]
        A = zeros(T, all, all*ensemble)
        a = zeros(T, all)
        val = Val(method)
        _init(val, A, a, hierarchy)
        new{T}(val, A, a)
    end
end

fit(m::LinearCombReconciler, hf::AbstractVector{<:HForecasts})::Nothing = _fit(m.method, m, hf)

function fit(m::LinearCombReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T})::Nothing where T<:ValueType 
    _fit(m.method, m, pred, obs, hierarchy)
end

function apply(m::LinearCombReconciler{T}, pred::AbstractMatrix{T})::AbstractMatrix{T} where T<:ValueType
    output = Matrix{T}(undef, size(pred, 1), size(m.recmatrix, 1))
    output .= pred*transpose(m.recmatrix)
    foreach(i -> output[:, i] .+= m.intercept[i], eachindex(m.intercept))
    return output
end

function apply(m::LinearCombReconciler{T}, hf::AbstractVector{<:HForecasts{T, I}})::AbstractVector{HForecasts{T, I}} where {T<:ValueType, I<:Integer}
    reconciled = apply(m, hcat((hf[i].pred for i in eachindex(hf))...))
    return HForecasts(reconciled, hf.obs, hf.id, hf.hierarchy)
end

function (m::LinearCombReconciler{T})(hf::AbstractVector{<:HForecasts{T, I}})::HForecasts{T, I} where {T<:ValueType, I<:Integer}
    apply(m, hf)
end

function (m::LinearCombReconciler{T})(pred::AbstractMatrix{T})::AbstractMatrix{T} where T<:ValueType
    apply(m, pred)
end
