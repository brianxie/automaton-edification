include("../datasets/mnist.jl")
include("neural_networks.jl")
include("activation_functions.jl")
include("../optimization/loss_functions.jl")

using Random, Statistics

training_samples = MNIST.read_file_from_index(MNIST.TRAINING_SAMPLES_PATH) .|> vec
training_labels = MNIST.read_file_from_index(MNIST.TRAINING_LABELS_PATH) .|> vec
test_samples = MNIST.read_file_from_index(MNIST.TEST_SAMPLES_PATH) .|> vec
test_labels = MNIST.read_file_from_index(MNIST.TEST_LABELS_PATH) .|> vec

"""
    rescale(sample, min, max)

Rescales `sample` such that it lies between `min` and `max`.
"""
function rescale(sample::AbstractVector, min::Number, max::Number)
    return (sample .- min) ./ (max - min)
end

"""
    one_hot(class_index, num_classes)

Creates a one-hot encoding of a value.
"""
function one_hot(class_index::Integer, num_classes::Integer)
    hot = zeros(num_classes)
    hot[class_index] = 1.0
    return hot
end

"""
    get_epoch(samples, labels)

Gets an epoch of training data.

Returns the samples and labels in a shuffled, normalized, one-hotted format.
"""
function get_epoch(samples::AbstractVector, labels::AbstractVector)
    rng = MersenneTwister(rand(UInt32))
    indices = randperm(rng, length(samples))

    m = mean(samples)

    processed_samples = samples[indices] .|>
        sample -> rescale(sample - m, 0, 255)

    processed_labels = labels[indices] .|>
        label -> one_hot(label[1]+1, 10)

    return processed_samples, processed_labels
end

"""
    check_accuracy(nn, samples, labels)

Runs prediction on all `samples` and `labels` using `nn`, and prints the
fraction of correct predictions.
"""
function check_accuracy(nn::NeuralNetworks.NeuralNetwork,
                        samples::AbstractVector,
                        labels::AbstractVector)
    counter = 0
    for i in 1:length(samples)
        if argmax(NeuralNetworks.predict(nn, samples[i])) ==
            argmax(labels[i])
            counter += 1
        end
    end
    print("$(counter) / $(length(samples))\n")
end

"""
    check_test_accuracy(nn)

Prints the fraction of correct predictions of `nn` against the test set.
"""
function check_test_accuracy(nn::NeuralNetworks.NeuralNetwork)
    processed_test_samples, processed_test_labels =
        get_epoch(test_samples, test_labels)

    check_accuracy(nn, processed_test_samples, processed_test_labels)
end

processed_training_samples, processed_training_labels =
    get_epoch(training_samples, training_labels)

layer_1 = NeuralNetworks.create_layer(784, 200, ActivationFunctions.relu)
layer_2 = NeuralNetworks.create_layer(200, 10, ActivationFunctions.softmax)
nn = NeuralNetworks.compose_layers(784, 10, LossFunctions.cross_entropy,
                                   layer_1, layer_2)

VECTORIZED = true

learning_rate = 0.03
batch_size = 20

@time begin
    stats = VECTORIZED ?
        NeuralNetworks.train_vectorized!(nn,
                                         processed_training_samples,
                                         processed_training_labels,
                                         learning_rate,
                                         batch_size) :
        NeuralNetworks.train!(nn,
                              processed_training_samples,
                              processed_training_labels,
                              learning_rate,
                              batch_size)
end # @time
