include("neural_networks.jl")

# TODO: make this work for batches
using LinearAlgebra, Random, Plots

learning_rate = 0.1
sq_error(x,y) = (x-y) .* (x-y)
e_approx = float(MathConstants.e)
sigmoid(x) = 1.0 / (1.0 + e_approx^(-x))

# Learn XOR
# [a, b] -> [a XOR b]
nn = NeuralNetworks.create_nn([2,1], sigmoid, sq_error, 2, 1)
EPOCHS = 25000
xs = 1:4*EPOCHS
ys = Vector(undef, 4*EPOCHS)
for (index, input) in enumerate(shuffle(repeat([[0,0],[0,1],[1,0],[1,1]], EPOCHS)))
    label = float(input[1] ⊻ input[2])

    # Run inference
    layer_1_forward = NeuralNetworks.forward_pass(nn.layers[1], input)
    layer_2_forward =
        NeuralNetworks.forward_pass(nn.layers[2], layer_1_forward.activations)

    # Compute loss
    loss = NeuralNetworks.compute_loss(layer_2_forward.activations,
                                      [label],
                                      sq_error)[1]

    # Backprop
    loss_grad = NeuralNetworks.compute_loss_gradient(layer_2_forward.activations,
                                                     [label],
                                                     sq_error)
    layer_2_backward =
        NeuralNetworks.compute_weight_gradients(nn.layers[2],
                                                layer_2_forward,
                                                loss_grad)
    layer_1_backward =
        NeuralNetworks.compute_weight_gradients(nn.layers[1],
                                                layer_1_forward,
                                                layer_2_backward.dL_dI[1:end-1])

    # Update weights
    nn.layers[2] = NeuralNetworks.update_layer_weights(nn.layers[2],
                                                       layer_2_backward,
                                                       learning_rate)
    nn.layers[1] = NeuralNetworks.update_layer_weights(nn.layers[1],
                                                       layer_1_backward,
                                                       learning_rate)
    ys[index] = loss
    if (index % 1000 == 0)
        print("Iteration $index / $(4 * EPOCHS)\n")
    end
end

plt = plot(xs[1:100:end], ys[1:100:end], seriestype = :scatter)
display(plt)

for input in [[0,0],[0,1],[1,0],[1,1]]
    print("Input: $input / " *
          "Prediction: $(NeuralNetworks.predict(nn, input)) / " *
          "Actual: $(input[1] ⊻ input[2])\n")
end
