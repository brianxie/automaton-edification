using CSV, DataFrames

include("../cluster.jl")
include("centroid_linear_classifier.jl")

SYN_CENTER_RADIUS = 100
SYN_CENTER_PERTURB_RADIUS = 10.0
SYN_VECTOR_NDIMS = 4
NUM_POSITIVE_SYN_POINTS = 100
NUM_NEGATIVE_SYN_POINTS = 100

function write_clusters_to_csv(ndims::Integer,
                               num_positive_points::Integer,
                               num_negative_points::Integer)
    # Generate two clusters of random data points, where each coordinate is an
    # integer in the range [-SYN_CENTER_RADIUS, SYN_CENTER_RADIUS].
    positive_center = rand(-SYN_CENTER_RADIUS:SYN_CENTER_RADIUS, ndims)
    negative_center = rand(-SYN_CENTER_RADIUS:SYN_CENTER_RADIUS, ndims)

    # Generate clusters of points uniformly distributed around each center.
    positive_cluster =
        Cluster.create_uniform_cluster(positive_center,
                                       SYN_CENTER_PERTURB_RADIUS,
                                       num_positive_points)
    negative_cluster =
        Cluster.create_uniform_cluster(negative_center,
                                       SYN_CENTER_PERTURB_RADIUS,
                                       num_negative_points)

    CSV.write("positive_data.csv", DataFrame(positive_cluster),
              writeheader=false)
    CSV.write("negative_data.csv", DataFrame(negative_cluster),
              writeheader=false)

    return positive_center, negative_center
end

function read_clusters_from_csv()
    positive_data =
        Matrix(CSV.read("positive_data.csv", header=false, DataFrame))
    negative_data =
        Matrix(CSV.read("negative_data.csv", header=false, DataFrame))

    return positive_data, negative_data
end

positive_center, negative_center =
    write_clusters_to_csv(SYN_VECTOR_NDIMS,
                          NUM_POSITIVE_SYN_POINTS, NUM_NEGATIVE_SYN_POINTS)
positive_data, negative_data = read_clusters_from_csv()

positive_centroid = Cluster.compute_centroid(positive_data)
negative_centroid = Cluster.compute_centroid(negative_data)

# Train the model.
classify = CentroidLinearClassifier.model(positive_data, negative_data)

println("Classifier trained!")
println("`positive_center`, `negative_center` are the original unperturbed centers.")
println("`positive_centroid`, `negative_centroid` are the respective perturbed centroids.")
println("`positive_data`, `negative_data` contain the full datasets.")
println("`classify` is a function that can be applied to a vector to classify it.")
println("The vector must be of dimension: ", SYN_VECTOR_NDIMS)
println()
println("positive_centroid: ", positive_centroid)
println("negative_centroid: ", negative_centroid)
