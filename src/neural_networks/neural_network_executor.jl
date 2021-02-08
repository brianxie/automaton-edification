include("neural_networks.jl")

using Statistics, LinearAlgebra, Random, Plots

VECTORIZED = false

# Parameters
batch_size = 32
learning_rate = 0.2
epochs = 64000
# Takes squared error along columns (dimensions), and average along rows
# (count); applicable to vectors or matrices.
mse(x,y) = mean(sum((x .- y) .^2, dims=2), dims=1)[1]
e_approx = float(MathConstants.e)
sigmoid_scalar(x) = (1.0 + MathConstants.e^(-x))^(-1)
sigmoid_vec(x) = sigmoid_scalar.(x)

nn = NeuralNetworks.create_nn([2,1], sigmoid_vec, mse, 2, 1)

# Learn XOR
# [a, b] -> [a XOR b]
data = shuffle(repeat([[0,0],[0,1],[1,0],[1,1]], epochs))
# Each label is a vector (even if a singleton)
labels = map(point -> [float(point[1] ⊻ point[2])], data)

@time begin
    stats = VECTORIZED ?
        NeuralNetworks.train_vectorized!(nn, data, labels, learning_rate, batch_size) :
        NeuralNetworks.train!(nn, data, labels, learning_rate, batch_size)
end

for input in [[0,0],[0,1],[1,0],[1,1]]
    print("Input: $input / " *
          "Prediction: $(NeuralNetworks.predict(nn, input)) / " *
          "Actual: $(input[1] ⊻ input[2])\n")
end

plt = plot(1:100:length(stats.losses),
           stats.losses[1:100:end],
           seriestype = :scatter)
display(plt)
