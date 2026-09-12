abstract type AbstractHierarchy{T<:ValueType} end

"""
    reconstruct(h::AbstractHierarchy, bottom::AbstractMatrix)
Reconstruct the hierarchical series from the bottom-level series.
"""
function reconstruct(h::AbstractHierarchy, bottom::AbstractMatrix)
    error("`reconstruct` is not implemented for the given hierarchy type")
end

"""
    check(h::AbstractHierarchy, obs::AbstractMatrix; tol=1e-4, warn=true)::Bool
Check if the hierarchical structure of `h` correctly reconstructs the original series `obs` with a specified tolerance.
"""
function check(h::AbstractHierarchy, obs::AbstractMatrix; tol::AbstractFloat=1e-4, warn::Bool=true)::Bool
    error("`check` is not implemented for the given hierarchy type")
end


function Base.convert(::Type{T1}, h::AbstractHierarchy{T2}) where {T1<:ValueType, T2<:ValueType}
    error("`convert` is not implemented for the given hierarchy type")    
end

function Base.size(h::AbstractHierarchy)
    error("`size` is not implemented for the given hierarchy type")
end


"""
    Hierarchy
Create a `Hierarchy` structure that encapsulates the summing matrix `S` defining the hierarchical relationships between series. 

The summing matrix should have dimensions `(all, bottom)`, where `all` is the total number of series in the hierarchy and `bottom` is the number of bottom-level series.

Hierarchy can be constructed in two ways:
- `Hierarchy(S::AbstractMatrix)` - directly from a summing matrix `S`
- `Hierarchy(obs::AbstractMatrix, bottom::Integer)` - by inferring the summing matrix from the observations `obs` and the number of bottom levels
"""
struct Hierarchy{T<:ValueType} <: AbstractHierarchy{T}
    S::AbstractMatrix{T}
    
    function Hierarchy(S::AbstractMatrix{T}) where {T<:ValueType}
        check_bottom_level(S)
        new{T}(AbstractMatrix{T}(S))
    end
end

function Hierarchy(obs::AbstractMatrix{T}, bottom::Integer) where {T<:ValueType}
    S = infer_summing_matrix(obs, bottom)
    h = Hierarchy(S)
    check(h, obs)
    return h
end

function Base.convert(::Type{T1}, h::Hierarchy{T2}) where {T1<:ValueType, T2<:ValueType}
    return Hierarchy(convert(Matrix{T1}, h.S))
end

function Base.size(h::Hierarchy)
    return size(h.S)
end

function reconstruct(h::Hierarchy, bottom_series::AbstractMatrix)
    return bottom_series*transpose(h.S)
end

function check(h::Hierarchy, obs::AbstractMatrix; tol::AbstractFloat=1e-4, warn::Bool=true)::Bool
    bottom = size(h)[end]
    @views reconstructed = reconstruct(h, obs[:, 1:bottom])
    incoherency = maximum(abs.(obs .- reconstructed))
    if incoherency > tol 
        warn && @warn "the summing matrix does not reconstruct the original series within a tolerance of $tol - incoherency: $incoherency"
        return false
    end
    return true
end
