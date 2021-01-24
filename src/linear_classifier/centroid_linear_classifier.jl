module CentroidLinearClassifier

include("../cluster.jl")

using LinearAlgebra

"""
    model(positive_data, negative_data)

Trains a binary classification model and returns a function that classifies a
point, where positive and negative results indicate that the point is positively
or negatively classified, respectively.
"""
function model(positive_data::AbstractMatrix{<:Number},
               negative_data::AbstractMatrix{<:Number})::Function
    @assert size(positive_data, 2) == size(negative_data, 2)

    positive_data_centroid = Cluster.compute_centroid(positive_data)
    negative_data_centroid = Cluster.compute_centroid(negative_data)

    normal_vector = positive_data_centroid .- negative_data_centroid
    midpoint = (positive_data_centroid .+ negative_data_centroid) ./ 2

    decision_function =
        function (x)
            return dot(normal_vector, x) - dot(normal_vector, midpoint)
        end

    return decision_function
end

end # module
