module AutomatonEdification

include("datasets/mnist.jl")

include("clusters/clusters.jl")

include("optimization/loss_functions.jl")

include("linear_classifiers/linear_classifiers.jl")
include("linear_classifiers/centroid_linear_classifier_impl.jl")
#=
Executables:
include("src/linear_classifiers/centroid_linear_classifier_executable.jl")
=#

include("neural_networks/activation_functions.jl")
include("neural_networks/neural_networks.jl")
#=
Executables:
include("src/neural_networks/xor_neural_network_executable.jl")
include("src/neural_networks/mnist_neural_network_executable.jl")
=#

end # module
