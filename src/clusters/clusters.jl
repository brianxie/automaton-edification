module Clusters

export create_uniform_cluster_ncube, create_uniform_cluster_nsphere, compute_centroid

using LinearAlgebra, Plots

"""
    create_uniform_cluster_ncube(center, perturbation, num_points)

Creates a cluster of size `num_points`, where each point is uniformly randomly
selected from the volume of an n-cube centered at `center`, aligned with the
standard basis, and with edge length 2 * `perturbation`

Returns a `num_points` * length(`center`) matrix, where each row is a point in
the cluster.
"""
function create_uniform_cluster_ncube(center::AbstractVector{<:Number},
                                      perturbation::Number,
                                      num_points::Integer)::AbstractMatrix{<:AbstractFloat}
    # (num_points, 1)
    ones_vec = ones(Integer, (num_points, 1))
    # (1, length(center))
    center_T = transpose(center)
    # (num_points, length(center)), where each row is a copy of center
    center_vecs = ones_vec * center_T

    # Generate uniformly random values between [0, perturbation) and negate some
    # randomly in order to avoid floating point precision errors and preserve
    # symmetry of the distribution.
    random_perturbations =
        perturbation * rand(Float64, (num_points, length(center)))
    random_flips = rand([-1, 1], (num_points, length(center)))

    return center_vecs + (random_perturbations .* random_flips)
end

"""
    create_uniform_cluster_nsphere(center, perturbation, num_points)

Creates a cluster of size `num_points`, centered around `center`, where each
point is uniformly randomly selected from the volume of an n-sphere with radius
'perturbation'.

Returns a `num_points` * length(`center`) matrix, where each row is a point in
the cluster.
"""
function create_uniform_cluster_nsphere(center::AbstractVector{<:Number},
                                        perturbation::Number,
                                        num_points::Integer)::AbstractMatrix{<:AbstractFloat}
    # (num_points, 1)
    ones_vec = ones(Integer, (num_points, 1))
    # (1, length(center))
    center_T = transpose(center)
    # (num_points, length(center)), where each row is a copy of center
    center_vecs = ones_vec * center_T

    # num_points * length(center) standard normal random variables
    rand_vecs = randn(Float64, (num_points, length(center)))
    # Take the norm along each column.
    # num_points
    norms = map(LinearAlgebra.norm,
                [rand_vecs[n,:] for n in 1:size(rand_vecs, 1)])
    # num_points uniform in [0, 1)
    uniform_coeffs = rand(Float64, num_points)

    origin_cluster = perturbation *
        Diagonal(uniform_coeffs .^ (1/length(center))) *
        Diagonal(1 ./ norms) *
        rand_vecs

    return center_vecs + origin_cluster
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
    return point_sum ./ size(points, 1)
end

"""
    plot_cluster(points)

Plots a matrix of points, where each row represents the coordinates of a single
point.
"""
function plot_cluster(points::AbstractMatrix{<:Number})
    plt = plot([points[:,n] for n in 1:size(points, 2)]...,
               seriestype = :scatter,
               showaxis = :show,
               aspect_ratio = :equal)
    display(plt)
end

end # module
