"""
    HForecasts(pred::AbstractMatrix, obs::AbstractMatrix, id::AbstractVector[, hierarchy])
Create an `HForecasts` structure storing the hierarchical series of `pred`ictions along with the `obs`ervations and `id`entifiers.

The optional `hierarchy` argument can be provided in three ways:
- hierarchical structure `::Hierarchy`
- summing matrix `::Matrix`
- number of bottom levels `::Integer` (the hierarchy will be inferred from the data)

The shape of `pred` and `obs` should be such that `[t, i]` corresponds to the value of the `i`-th series at time `t`.
"""
struct HForecasts{T<:ValueType, I<:Integer}
    pred::Matrix{T}
    obs::Matrix{T}
    id::Vector{I}
    hierarchy::AbstractHierarchy{T}

    function HForecasts(pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, id::AbstractVector{I}, hierarchy::AbstractHierarchy{T}) where {T<:ValueType, I<:Integer}
        @assert size(pred) == size(obs) "the size of `pred` must match the size of `obs`"
        @assert length(id) == size(obs, 1) "the length of `id` must match the number of rows in `obs`"
        @assert isunique(id) "`id` must contain only unique elements"
        @assert size(hierarchy)[begin] == size(obs, 2) "the summing matrix does not match the number of series"
        new{T, I}(Matrix{T}(pred), Matrix{T}(obs), Vector{I}(id), hierarchy)
    end
end

# promoting constructor
function HForecasts(pred::AbstractMatrix{T1}, obs::AbstractMatrix{T2}, id::AbstractVector{<:Integer}, hierarchy::AbstractHierarchy{T3}) where {T1<:ValueType, T2<:ValueType, T3<:ValueType}
    T = promote_type(T1, T2, T3)
    HForecasts(convert(Matrix{T}, pred), convert(Matrix{T}, obs), id, convert(T, hierarchy))
end

# promoting constructor with summing matrix 
function HForecasts(pred::AbstractMatrix{T1}, obs::AbstractMatrix{T2}, id::AbstractVector{<:Integer}, S::Matrix{<:Number}) where {T1<:ValueType, T2<:ValueType}
    T = promote_type(T1, T2)
    hierarchy = Hierarchy(convert(Matrix{T}, S))
    HForecasts(convert(Matrix{T}, pred), convert(Matrix{T}, obs), id, hierarchy)
end

# promoting constructor with inferred summing matrix
function HForecasts(pred::AbstractMatrix{T1}, obs::AbstractMatrix{T2}, id::AbstractVector{<:Integer}, bottom_levels::Integer) where {T1<:ValueType, T2<:ValueType}
    T = promote_type(T1, T2)
    obs = convert(Matrix{T}, obs)
    HForecasts(convert(Matrix{T}, pred), obs, id, Hierarchy(obs, bottom_levels))
end

function Base.length(hf::HForecasts)
    return length(hf.id)
end

function Base.getindex(hf::HForecasts, t::Integer)
    return (pred = hf.pred[t, :],
            obs = hf.obs[t, :],
            id = hf.id[t])
end

function Base.getindex(hf::HForecasts, T::AbstractVector{<:Integer})
    return HForecasts(
            @view(hf.pred[T, :]),
            @view(hf.obs[T, :]),
            @view(hf.id[T]),
            hf.hierarchy)
end

function Base.firstindex(hf::HForecasts)
    return firstindex(hf.id)
end

function Base.lastindex(hf::HForecasts)
    return lastindex(hf.id)
end

function Base.eachindex(hf::HForecasts)
    return eachindex(hf.id)
end

function Base.size(hf::HForecasts)
    return size(hf.obs)
end

"""
    validate(hf::HForecasts, tol=1e-4)::Bool
Validate the consistency of `hf` by checking if the summing matrix correctly sums the observations. 

Returns `true` if the maximum absolute error across time and levels is less than `tol`, otherwise returns `false` and throws @warn.
"""
validate(hf::HForecasts, tol::AbstractFloat=1e-4)::Bool = check(hf.hierarchy, hf.obs, tol)
