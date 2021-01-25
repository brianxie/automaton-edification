module LinearClassifiers

export AbstractLinearClassifier, classify

"""
Supertype of all linear classifiers.

Each linear classifier should have the fields:
- `classify::Function`
"""
abstract type AbstractLinearClassifier end

"""
    classify(point, model)

Classifies `point` using the `classify` function of `model`.
"""
classify(point::AbstractVector{<:Number},
         model::AbstractLinearClassifier) = model.classify(point)

end # module
