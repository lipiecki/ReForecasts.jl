const ICombFamily = Union{Val{:icomb}, Val{:ridgeicomb}}
const MIN_LAMBDA_RATIO = 1e-6
const NLAMBDAS = 20
const NFOLDS = 7

function _init(::ICombFamily, recmatrix::AbstractMatrix{T}, intercept::AbstractVector{T}, _::Vararg) where {T<:ValueType}
    fill!(recmatrix, zero(T))
    fill!(intercept, zero(T))
    return nothing
end

function _fit(::Val{:icomb}, m::Union{LinearReconciler, LinearCombReconciler}, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, _::Vararg)::Nothing where T<:ValueType
    n, p = size(pred)
    X = hcat(pred, ones(T, n))

    XtX = Matrix{T}(undef, p+1, p+1)
    XtY = Matrix{T}(undef, p+1, p)
    mul!(XtX, X', X)
    mul!(XtY, X', obs)

    ldiv!(cholesky!(Symmetric(XtX)), XtY)

    transpose!(m.recmatrix, @view XtY[1:end-1, :])
    copyto!(m.intercept, @view XtY[end, :])
    return nothing
end


function ridge_lambda_path(X::AbstractMatrix{<:Number}, Y::AbstractMatrix{<:Number}; min_lambda_ratio::AbstractFloat=MIN_LAMBDA_RATIO, nlambdas::Integer=NLAMBDAS)
    lambda_max = sum(abs2, cov(X, Y), dims=2) |> maximum |> sqrt
    lambda_max /= size(X, 1)*1e-3
    lambda_min = lambda_max * min_lambda_ratio
    return exp.(range(log(lambda_max), log(lambda_min), length=nlambdas))
end

function _fit(::Val{:ridgeicomb}, m::Union{LinearReconciler, LinearCombReconciler}, pred::AbstractMatrix{T}, obs::AbstractMatrix{T}, _::Vararg)::Nothing where T<:ValueType
    n, p = size(pred)
    lambda = ridge_lambda_path(pred, obs)
    
    X = hcat(pred, ones(T, n))
    
    # random folds
    idx = randperm(n)
    fold_size = Int(floor(n / NFOLDS))

    # variance scaling matrix
    scale_vec = vec(std(X, dims=1))
    scale_vec[end] = zero(T)
    scale = Diagonal(scale_vec)

    best_mse = typemax(T)
    best_lambda_idx = 1
    coeffs = zeros(T, p + 1, size(obs, 2))

    for fold in 1:NFOLDS
        test_idx = idx[end-fold*fold_size+1 : end-(fold-1)*fold_size]
        train_idx = setdiff(idx, test_idx)
        
        @views Xtrain, Ytrain = X[train_idx, :], obs[train_idx, :]
        @views Xtest, Ytest   = X[test_idx, :],  obs[test_idx, :]

        XtX = Xtrain' * Xtrain
        Xty = Xtrain' * Ytrain

        for i in eachindex(lambda)
            A = Symmetric(XtX + lambda[i] .* scale)
            coeffs .= A \ Xty
            
            mse = mean(abs2, Ytest .- Xtest * coeffs)
            if mse < best_mse
                best_mse = mse
                best_lambda_idx = i
            end
        end
    end

    XtX_full = X' * X
    Xty_full = X' * obs
    A_full = Symmetric(XtX_full + lambda[best_lambda_idx] .* scale)
    
    coeffs .= A_full \ Xty_full

    transpose!(m.recmatrix, @view coeffs[1:end-1, :])
    copyto!(m.intercept, @view coeffs[end, :])
    return nothing
end

function _fit(::Val{:icomb}, m::LinearCombReconciler, hf::AbstractVector{<:HForecasts{T, I}})::Nothing where {T<:ValueType, I<:Integer}
    pred = hcat((hf[i].pred for i in eachindex(hf))...)
    obs = hf[begin].obs
    _fit(Val(:icomb), m, pred, obs)
end

function _fit(::Val{:ridgeicomb}, m::LinearCombReconciler, hf::AbstractVector{<:HForecasts{T, I}})::Nothing where {T<:ValueType, I<:Integer}
    pred = hcat((hf[i].pred for i in eachindex(hf))...)
    obs = hf[begin].obs
    _fit(Val(:ridgeicomb), m, pred, obs)
end
