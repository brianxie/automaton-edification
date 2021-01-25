module CentroidLinearClassifiers

export CentroidLinearClassifier, train_model

include("../cluster.jl")
include("linear_classifiers.jl")

using LinearAlgebra
using .LinearClassifiers, .Cluster

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
    train_model(positive_data, negative_data)

Trains a binary classification model.

The returned model has a field `classify`, a function that classifies a point
and returns a scalar value, where positive and negative results indicate that
the point is positively or negatively classified, respectively.
"""
function train_model(positive_data::AbstractMatrix{<:Number},
                     negative_data::AbstractMatrix{<:Number})::CentroidLinearClassifier
    @assert size(positive_data, 2) == size(negative_data, 2)

    positive_data_centroid = Cluster.compute_centroid(positive_data)
    negative_data_centroid = Cluster.compute_centroid(negative_data)

    normal_vector = positive_data_centroid .- negative_data_centroid
    midpoint = (positive_data_centroid .+ negative_data_centroid) ./ 2

    offset = -1 * dot(normal_vector, midpoint)

    decision_function =
        function (x)
            return dot(normal_vector, x) + offset
        end

    return CentroidLinearClassifier(decision_function, normal_vector, offset)
end

end # module
