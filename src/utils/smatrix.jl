function check_bottom_level(S::AbstractMatrix{<:Number})::Nothing
    all, bottom = size(S)
    @assert all > bottom "the number of all nodes must be greater than the number of bottom-level nodes"
    # check if the bottom block corresponds to the identity matrix
    for i in 1:bottom
        (S[i, i] ≈ 1 && sum(S[i, j] for j in 1:bottom) ≈ 1) || error("the bottom block of the summing matrix must be an identity matrix")
    end
    return nothing
end

function infer_summing_matrix(obs::AbstractMatrix{T}, bottom::Integer, tol::AbstractFloat=1e-4)::Matrix{T} where {T<:Number} 
    all = size(obs, 2)
    S = zeros(T, all, bottom)
    @views X = obs[:, 1:bottom]
    S .= transpose(X\obs)
    return S
end
