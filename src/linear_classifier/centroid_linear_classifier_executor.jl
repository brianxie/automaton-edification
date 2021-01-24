using CSV, DataFrames

include("../cluster.jl")
include("centroid_linear_classifier.jl")

vector_dimension = 3
positive_size = 100
negative_size = 100

function write_clusters_to_csv(dimensions::Int64,
                               positive_size::Int64,
                               negative_size::Int64)
    # Generate two clusters random data points (magic multiplicative factor of
    # 32).
    positive_center = (rand(Float64, dimensions) .- 0.5) * 32
    negative_center = (rand(Float64, dimensions) .- 0.5) * 32

    # Generate clusters of points uniformly distributed around each center
    # (magic radius of 1.0).

    positive_cluster =
        Cluster.create_uniform_cluster(positive_center, 1.0, positive_size)
    negative_cluster =
        Cluster.create_uniform_cluster(negative_center, 1.0, negative_size)

    CSV.write("positive_data.csv",
              DataFrame(positive_cluster), writeheader=false)
    CSV.write("negative_data.csv",
              DataFrame(negative_cluster), writeheader=false)

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
    write_clusters_to_csv(vector_dimension, positive_size, negative_size)
positive_data, negative_data = read_clusters_from_csv()

positive_centroid = Cluster.compute_centroid(positive_data)
negative_centroid = Cluster.compute_centroid(negative_data)

# Train the model.
model = CentroidLinearClassifier.model(positive_data, negative_data)

println("Model trained!")
println("`positive_data`, `negative_data` contain the full datasets.")
println("`positive_centroid`, `negative_centroid` are the respective centroids.")
println("`model` is a function that can be applied to a vector to classify it.")
