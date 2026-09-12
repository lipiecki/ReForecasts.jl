abstract type Reconciler end

_fit(method::Val, args::Vararg)::Nothing = error("unsupported reconciliation method")

_init(method::Val, args::Vararg)::Nothing = error("unsupported reconciliation method")

"""
    LinearReconciler
Create a `LinearReconciler` structure that performs linear reconciliation of hierarchical forecasts.

`LinearReconciler(method::Symbol, hierarchy::Hierarchy{T})` constructs a reconciler for the specified `method` and hierarchy. The `method` can be one of the following:
- `:bu` - Bottom-Up reconciliation
- `:ols` - Ordinary Least Squares reconciliation
- `:mint` - Minimum Trace reconciliation [(Wickramasuriya et al. 2019)](https://doi.org/10.1080/01621459.2018.1448825)
- `:shrinkmint` - Minimum Trace reconciliation with covariance shrinkage by [Schäfer & Strimmer (2005)](https://doi.org/10.2202/1544-6115.1175)
- `:honeymint` - Minimum Trace reconciliation with covariance shrinkage by [Ledoit & Wolf (2004)](https://doi.org/10.3905/jpm.2004.110)
- `:icomb` - Information Combination [(Nguyen et al. 2026)](https://arxiv.org/abs/2605.29611)
- `:ridgeicomb` - Information Combination with ridge regularization [(Nguyen et al. 2026)](https://arxiv.org/abs/2605.29611)
"""
struct LinearReconciler{T<:ValueType} <: Reconciler
    method::Val
    recmatrix::AbstractMatrix{T}
    intercept::AbstractVector{T}

    function LinearReconciler(method::Symbol, A::AbstractMatrix{T}, a::AbstractVector{T}) where T<:ValueType
        new{T}(Val(method), A, a)
    end

    function LinearReconciler(method::Symbol, hierarchy::Hierarchy{T}) where T<:ValueType
        all = size(hierarchy)[begin]
        A = zeros(T, all, all)
        a = zeros(T, all)
        val = Val(method)
        _init(val, A, a, hierarchy)
        new{T}(val, A, a)
    end
end

fit(m::LinearReconciler, hf::HForecasts)::Nothing = _fit(m.method, m, hf.pred, hf.obs, hf.hierarchy)

fit(m::LinearReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T}) where T<:ValueType = _fit(m.method, m, pred, obs, hierarchy)

fit(m::LinearReconciler, errors::AbstractMatrix{T}, hierarchy::Hierarchy{T}) where T<:ValueType = _fit(m.method, m, errors, errors.*0, hierarchy)

function apply!(m::LinearReconciler{T}, pred::AbstractMatrix{T})::Nothing where T<:ValueType
    pred .= pred*transpose(m.recmatrix)
    foreach(i -> pred[:, i] .+= m.intercept[i], eachindex(m.intercept))
    return nothing
end

function apply!(m::LinearReconciler{T}, hf::HForecasts{T, I})::Nothing where {T<:ValueType, I<:Integer}
    apply!(m, hf.pred)
    return nothing
end

function apply(m::LinearReconciler{T}, pred::AbstractMatrix{T})::AbstractMatrix{T} where T<:ValueType
    output = Matrix{T}(undef, size(pred))
    output .= pred
    apply!(m, output)
    return output
end

function apply(m::LinearReconciler{T}, hf::HForecasts{T, I})::HForecasts{T, I} where {T<:ValueType, I<:Integer}
    reconciled = apply(m, hf.pred)
    return HForecasts(reconciled, hf.obs, hf.id, hf.hierarchy)
end

function (m::LinearReconciler{T})(hf::HForecasts{T, I})::HForecasts{T, I} where {T<:ValueType, I<:Integer}
    apply(m, hf)
end

function (m::LinearReconciler{T})(pred::AbstractMatrix{T})::AbstractMatrix{T} where T<:ValueType
    apply(m, pred)
end

include("reconcilers/bu.jl")
include("reconcilers/icomb.jl")
include("reconcilers/mint.jl")
include("reconcilers/ols.jl")
