module Cluster

using LinearAlgebra

"""
    create_uniform_cluster(center, perturbation, num_points)

Create a uniformly randomly generated cluster of size `num_points`, centered
around `center`.

Each point in the cluster deviates from `center` by a uniformly random number in
the range [0, `perturbation`).

Returns a `num_points` * length(`center`) matrix, where each row is a point in
the cluster.
"""
function create_uniform_cluster(center::AbstractVector{<:Number},
                                perturbation::Number,
                                num_points::Integer)::AbstractMatrix{<:AbstractFloat}
    # (num_points, 1)
    ones_vec = ones(Integer, (num_points, 1))
    # (1, length(center))
    center_T = transpose(center)

    # (num_points, length(center)), where each row is a copy of center
    center_vecs = ones_vec * center_T

    # Generate random values between [0, perturbation) and negate some randomly
    # in order to avoid floating point precision errors and preserve symmetry of
    # the distribution.
    random_perturbations =
        perturbation * rand(Float64, (num_points, length(center)))
    random_flips = rand([-1, 1], (num_points, length(center)))

    return center_vecs + (random_perturbations .* random_flips)
end

"""
    compute_centroid(points)

Computes the centroid of a matrix of points.

`points` is an m * n matrix that describes m points (rows) of dimension n
(columns).

Returns a single vector of size `n` corresponding to the centroid.
"""
function compute_centroid(
        points::AbstractMatrix{<:Number})::AbstractVector{<:AbstractFloat}
    point_sum = sum(points, dims=1)[:]
    return point_sum ./ length(points)
end

end # module
