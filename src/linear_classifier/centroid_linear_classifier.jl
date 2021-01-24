module CentroidLinearClassifier

include("../cluster.jl")

using LinearAlgebra

function model(cs::Array{Float64, 2}, xs::Array{Float64, 2})::Function
    @assert size(cs, 2) == size(xs, 2)
    cs_centroid = Cluster.compute_centroid(cs)
    xs_centroid = Cluster.compute_centroid(xs)

    normal_vector = cs_centroid .- xs_centroid
    midpoint = (cs_centroid .+ xs_centroid) ./ 2

    decision_function =
        function (x)
            return dot(normal_vector, x) - dot(normal_vector, midpoint)
        end

    return decision_function
end

end
