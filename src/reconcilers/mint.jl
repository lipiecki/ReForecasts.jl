const MinTFamily = Union{Val{:mint}, Val{:shrinkmint}, Val{:honeymint}, Val{:wls}}

function _init(::MinTFamily, recmatrix::AbstractMatrix{T}, intercept::AbstractVector{T}, _::Vararg) where {T<:ValueType}
    fill!(recmatrix, zero(T))
    fill!(intercept, zero(T))
    return nothing
end

function _reconcile_mint!(recmatrix::AbstractMatrix{T}, covmatrix::AbstractMatrix{T}, S::AbstractMatrix{T}) where {T<:ValueType}
    _, nbottom = size(S)
    
    # P = J − JWU(U'WU)^(-1)U'
    # P = [(I+KC) -K]
    # K = (W12 - W11C')(U'WU)^(-1)

    C = @view S[nbottom+1:end, :]
    W11 = @view covmatrix[1:nbottom, 1:nbottom]
    W21 = @view covmatrix[nbottom+1:end, 1:nbottom]
    W22 = @view covmatrix[nbottom+1:end, nbottom+1:end]

    Kt = Matrix{T}(W21)
    mul!(Kt, C, W11, -one(T), one(T))

    UtWU = Matrix{T}(W22)
    mul!(UtWU, W21, C', -one(T), one(T))
    mul!(UtWU, C, Kt', -one(T), one(T))

    ldiv!(cholesky!(Symmetric(UtWU)), Kt)

    rec_right = @view recmatrix[:, nbottom+1:end]
    mul!(rec_right, S, Kt', -one(T), zero(T))

    rec_left = @view recmatrix[:, 1:nbottom]
    copyto!(rec_left, S)
    mul!(rec_left, rec_right, C, -one(T), one(T))
end

function _fit(::Val{:mint}, m::LinearReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T})::Nothing where T<:ValueType
    errors = obs .- pred
    covmatrix = transpose(errors)*errors ./ size(errors, 2)

    _reconcile_mint!(m.recmatrix, covmatrix, hierarchy.S)
    fill!(m.intercept, 0)
    return nothing
end

function _fit(::Val{:wls}, m::LinearReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T})::Nothing where T<:ValueType
    errors = obs .- pred
    covmatrix = Diagonal(transpose(errors)*errors ./ size(errors, 2))

    _reconcile_mint!(m.recmatrix, covmatrix, hierarchy.S)
    fill!(m.intercept, 0)
    return nothing
end

function _fit(::Val{:shrinkmint}, m::LinearReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T})::Nothing where T<:ValueType
    errors = obs .- pred
    n, d = size(errors)
    covmatrix = transpose(errors)*errors ./ n
    target = Diagonal(covmatrix)

    nominator = zero(T)     # variance of correlation estimates
    denominator = zero(T)   # squares of correlation estimates
    for j in 1:d
        for i in j+1:d
            scaleij = sqrt(covmatrix[i, i]*covmatrix[j, j])
            corrij = covmatrix[i, j]/scaleij
            @views nominator += sum(abs2, errors[:, i].*errors[:, j]./scaleij .- corrij) / n
            denominator += abs2(corrij)
        end
    end
    λ = max(0, min(1, nominator / denominator / n))
    covmatrix .= (1 - λ).*covmatrix .+ λ.*target
    _reconcile_mint!(m.recmatrix, covmatrix, hierarchy.S)
    fill!(m.intercept, 0)
    return nothing
end

function _fit(::Val{:honeymint}, m::LinearReconciler, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, hierarchy::Hierarchy{T})::Nothing where T<:ValueType
    errors = obs .- pred
    n, d = size(errors)
    covmatrix = transpose(errors)*errors ./ n
    corrmatrix = covmatrix ./ sqrt.(diag(covmatrix)*transpose(diag(covmatrix)))
    target = zeros(T, d, d)
    target[diagind(target)] .= diag(covmatrix)

    meancorr = zero(T)
    for j in 1:d
        for i in j+1:d
            meancorr += corrmatrix[i, j]
        end
    end
    meancorr /= d*(d-1)/2
    
    nominator = zero(T)
    denominator = zero(T)
    for j in 1:d
        for i in j+1:d
            covij = covmatrix[i, j]
            covii = covmatrix[i, i]
            covjj = covmatrix[j, j]
            scaleij = sqrt(covii*covjj)
            @views fij = 0.5 .* sum(
                (errors[:, i].*errors[:, j] .- covij) .*
                ( (abs2.(errors[:, i]) .- covii).*sqrt(covjj/covii) .+ (abs2.(errors[:, j]) .- covjj).*sqrt(covii/covjj) )
            ) / n
            @views nominator += sum(abs2, errors[:, i].*errors[:, j] .- covij)/n - meancorr*fij
            denominator += abs2(covij - meancorr*scaleij)
            target[i, j] = meancorr*scaleij
            target[j, i] = target[i, j]
        end
    end
    λ = max(0, min(1, nominator / denominator / n))
    covmatrix .= (1 - λ).*covmatrix .+ λ.*target
    _reconcile_mint!(m.recmatrix, covmatrix, hierarchy.S)
    fill!(m.intercept, 0)
    return nothing
end
