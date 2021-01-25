module LinearClassifiers

export AbstractLinearClassifier, CentroidLinearClassifier, classify

"""
Supertype of all linear classifiers.

Each linear classifier should have the fields:
- `classify::Function`
"""
abstract type AbstractLinearClassifier end

struct CentroidLinearClassifier <: AbstractLinearClassifier
    classify::Function
    # `coeffs` and `offset` are parameters for the linear decision boundary, for
    # a function of the form:
    # f(x) = (coeffs \dot x) + offset
    # where the decision boundary is defined by f(x) = 0
    coeffs::AbstractVector{<:Number}
    offset::Number
end

"""
    classify(point, model)

Classifies `point` using the `classify` function of `model`.
"""
classify(point::AbstractVector{<:Number},
         model::AbstractLinearClassifier) = model.classify(point)

end # module
