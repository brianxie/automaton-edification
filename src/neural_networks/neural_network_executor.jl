include("neural_networks.jl")

using LinearAlgebra, Random, Plots

# Parameters
learning_rate = 0.1
epochs = 25000
mse(x,y) = dot(x-y, x-y) / length(x)
e_approx = float(MathConstants.e)
sigmoid(x) = 1.0 / (1.0 + e_approx^(-x))

nn = NeuralNetworks.create_nn([2,1], sigmoid, mse, 2, 1)

# Learn XOR
# [a, b] -> [a XOR b]
data = shuffle(repeat([[0,0],[0,1],[1,0],[1,1]], epochs))
# Each label is a vector (even if a singleton)
labels = map(point -> [float(point[1] ⊻ point[2])], data)
stats = NeuralNetworks.train!(nn, data, labels, learning_rate)
plt = plot(1:100:length(stats.losses),
           stats.losses[1:100:end],
           seriestype = :scatter)
display(plt)

for input in [[0,0],[0,1],[1,0],[1,1]]
    print("Input: $input / " *
          "Prediction: $(NeuralNetworks.predict(nn, input)) / " *
          "Actual: $(input[1] ⊻ input[2])\n")
end
