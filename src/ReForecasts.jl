module ReForecasts

const ValueType = AbstractFloat

import Base: size, convert, eachindex, firstindex, getindex, lastindex, length, size
using LinearAlgebra, SparseArrays, Statistics, Random

include("utils/general.jl")
include("utils/smatrix.jl")
include("Hierarchy.jl")
include("HForecasts.jl")
include("CombReconciler.jl")
include("Reconciler.jl")
    
export 
    Hierarchy,
    convert,
    size,
    reconstruct,
    check,

    HForecasts,
    eachindex,
    firstindex,
    getindex,
    lastindex,
    length,
    validate,
    
    LinearReconciler,
    LinearCombReconciler,
    apply,
    apply!,
    fit
    
end # module ReForecasts
