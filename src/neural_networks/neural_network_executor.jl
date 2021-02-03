include("neural_networks.jl")

using Statistics, LinearAlgebra, Random, Plots

VECTORIZED = true

# Parameters
learning_rate = 0.1
epochs = 25000
mse(x,y) = mean(sum((x .- y) .^2, dims=2), dims=1)[1]
e_approx = float(MathConstants.e)
sigmoid(x) = 1.0 / (1.0 + e_approx^(-x))

# Used for vectorized implementation.
if VECTORIZED
    batch_size = 4
    learning_rate = 0.3
    epochs = 100
end

nn = NeuralNetworks.create_nn([2,1], sigmoid, mse, 2, 1)

# Learn XOR
# [a, b] -> [a XOR b]
data = shuffle(repeat([[0,0],[0,1],[1,0],[1,1]], epochs))
# Each label is a vector (even if a singleton)
labels = map(point -> [float(point[1] ⊻ point[2])], data)
# This can also be replaced with `train_vectorized!`.
stats = VECTORIZED ?
    NeuralNetworks.train_vectorized!(nn, data, labels, learning_rate, batch_size) :
    NeuralNetworks.train!(nn, data, labels, learning_rate)
plt = plot(1:100:length(stats.losses),
           stats.losses[1:100:end],
           seriestype = :scatter)
display(plt)

for input in [[0,0],[0,1],[1,0],[1,1]]
    print("Input: $input / " *
          "Prediction: $(NeuralNetworks.predict(nn, input)) / " *
          "Actual: $(input[1] ⊻ input[2])\n")
end
