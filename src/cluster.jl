module Cluster

using LinearAlgebra

# Returns a num_points * size(center) array of vectors
function create_uniform_cluster(center::Array{Float64, 1},
                                perturbation::Float64,
                                num_points::Int64)
    ones_vec = ones(Float64, num_points, 1)
    reshaped_center = transpose(reshape(center, size(center, 1), 1))

    # num_points * 1, 1 * size(center) => num_points copies of center
    center_vecs = ones_vec * reshaped_center

    random_perturbations =
        (rand(Float64, num_points, size(center, 1), ) .- 0.5) * 2

    return center_vecs + random_perturbations
end

function compute_centroid(points::Array{Float64, 2})::Array{Float64, 1}
    point_sum = sum(points, dims=1)[:]
    return point_sum ./ size(points, 1)
end

end # module
