module CentroidLinearClassifiers

include("../cluster.jl")
include("linear_classifiers.jl")

using LinearAlgebra
using .LinearClassifiers

struct CentroidLinearClassifier <: AbstractLinearClassifier
    classify::Function
    # TODO: decision boundary
    # f(x) = w dot x + alpha, where f(x) = 0
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

    decision_function =
        function (x)
            return dot(normal_vector, x) - dot(normal_vector, midpoint)
        end

    return CentroidLinearClassifier(decision_function)
end

end # module
